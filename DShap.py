
#______________________________________PEP8____________________________________
#_______________________________________________________________________
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf

import sys
from shap_utils import *
from Shapley import ShapNN
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score

class DShap(object):
    
    def __init__(self, X, y, X_test, y_test, num_test, sources=None, 
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, every points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            overwrite: Delete existing data and start computations from 
                scratch   %start from scratch 
            **kwargs: Arguments of the model
        """
            
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if overwrite and os.path.exists(directory):             % os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
                tf.compat.v1.gfile.DeleteRecursively(directory)           # 递归删除所有目录及其文件，directory即目录名，无返回。
            if not os.path.exists(directory):
                os.makedirs(directory)                          # os.makedirs(directory)   创建多层目录directory
                os.makedirs(os.path.join(directory, 'weights')) #在directory后拼接路径weights
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test, 
                                      sources, sample_weight)    #后边定义
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'  #assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        is_regression = (np.mean(self.y//1 == self.y) != 1)               #//取整除 - 返回商的整数部分（向下取整）
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
            
    def _initialize_instance(self, X, y, X_test, y_test, num_test, 
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""      
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(X))}  # 每个元素为i：np.array([i])
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            self._load_dataset(data_dir)
        else:
            self.X_heldout = X_test[:-num_test]
            self.y_heldout = y_test[:-num_test]
            self.X_test = X_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.Full_X_test = self.X_test
            self.X, self.y, self.sources = X, y, sources
            self.sample_weight = sample_weight
            data_dic = {'X': self.X, 'y': self.y, 'X_test': self.X_test,
                     'y_test': self.y_test, 'X_heldout': self.X_heldout,
                     'y_heldout':self.y_heldout, 'sources': self.sources}
            if sample_weight is not None:
                data_dic['sample_weight'] = sample_weight
                warnings.warn("Sample weight not implemented for G-Shapley")
            pkl.dump(data_dic, open(data_dir, 'wb'))        # pickle.dump(obj, file, protocol)将对象obj保存到文件file中去
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']  # pickle.load(file)反序列化对象，将文件中的数据解析为一个python对象。
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        n_points = len(self.X)
        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number,
            n_points, n_sources, self.model_family)
        
    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                   n_points, n_sources, model_family):
        tmc_dir = os.path.join(
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))  # zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0。
        )
        g_dir = os.path.join(
            directory, 
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points*self.X.shape[-1]))##长度是s*f
        self.mem_g = np.zeros((0, n_points*self.X.shape[-1]))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        if model_family not in ['logistic', 'NN']:
            return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))
        
    def _load_dataset(self, data_dir):
        '''Load the different sets of data if already exists.'''
        data_dic = pkl.load(open(data_dir, 'rb'))
        self.X_heldout = data_dic['X_heldout']
        self.y_heldout = data_dic['y_heldout']
        self.X_test = data_dic['X_test']
        self.y_test = data_dic['y_test']
        self.X = data_dic['X'] 
        self.y = data_dic['y']
        self.sources = data_dic['sources']
        if 'sample_weight' in data_dic.keys():
            self.sample_weight = data_dic['sample_weight']
        else:
            self.sample_weight = None
        
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)     # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                      for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1])
                     for name in previous_results if 'mem_g' in name]        
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0' 
        return tmc_number, g_number
    
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':          #F1分数（F1-Score），又称为平衡F分数（BalancedScore），它被定义为精确率和召回率的调和平均数。
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y, 
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data 
                different from test set.
            y: Labels, if valuation is performed on a data 
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')
        
    def run(self, save_every, err, tolerance=0.01, g_run=False, loo_run=False):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
        print('LOO values calculated!')
        tmc_run = True 
        g_run = g_run and self.model_family in ['logistic', 'NN']    #????
        while tmc_run or g_run:
            if g_run:
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
            if self.directory is not None:
                self.save_results()
            
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory, 
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )  
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))  
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance         
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            
            marginals, idxs = self.one_iteration(
                tolerance=tolerance, 
                sources=sources
            )
            
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            if self.sample_weight is None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(self.X, self.y,
                              sample_weight=self.sample_weight)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model, 
                    metric=self.metric,
                    X=self.Full_X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""   #TMC的核心算法
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        idxf = np.random.permutation(self.X.shape[-1])##取特征长度
        marginal_contribs = np.zeros((len(self.X),self.X.shape[-1]))##s*f维度
        temp_contribs = np.zeros((len(self.X),self.X.shape[-1]))
        for f, idxsf in enumerate(idxf):
            X_batch = np.zeros((0, f+1)) ##比如只需两个特征，我就要两列就行
            y_batch = np.zeros(0, int)
            self.X_test = self.Full_X_test[:, idxf[:f+1]]
            sample_weight_batch = np.zeros(0)
            truncation_counter = 0
            for s, idx in enumerate(idxs):
                new_data_X = [self.X[sources[idx], idxf[:f+1]]]
                new_data_y = self.y[sources[idx]]
                X_batch = np.concatenate((X_batch, new_data_X))##取某个样本的某几列,注意【】，这样才能得到正确的数据结构
                y_batch = np.concatenate((y_batch, new_data_y))
                #print(y_batch)
                if self.sample_weight is None:
                    sample_weight_batch = None
                else:
                    sample_weight_batch = np.concatenate([
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
                if(sum(self.y[idxs[:s+1]]==0)==(s+1)):
                    temp_contribs[sources[idx], idxsf] = sum(self.y_test==0)/len(self.y_test)
                    temp_contribs[sources[idx], idxsf] /= len(sources[idx])
                    continue   
                if(sum(self.y[idxs[:s+1]]==1)==(s+1)):
                    temp_contribs[sources[idx], idxsf] = sum(self.y_test==1)/len(self.y_test)
                    temp_contribs[sources[idx], idxsf] /= len(sources[idx])
                    continue
                                                ##如果一个样本，对于分类没有意义
                self.restart_model()
                if sample_weight_batch is None:
                    #print('fit')
                    self.model.fit(X_batch, y_batch)
                else:
                    self.model.fit(
                        X_batch, 
                        y_batch,
                        sample_weight = sample_weight_batch
                                )
                            
                       
                temp_contribs[sources[idx], idxsf] = self.value(self.model, metric=self.metric)
                temp_contribs[sources[idx], idxsf] /= len(sources[idx])
                distance_to_full_score = np.abs(temp_contribs[sources[idx], idxsf] - self.mean_score)
                # if distance_to_full_score <= tolerance * self.mean_score and idx>0:
                #     temp_contribs[sources[idx], idxsf] = temp_contribs[sources[idx-1], idxsf]
                #     truncation_counter += 1
                #     if truncation_counter > 5:
                #         break
                # else:
                #     truncation_counter = 0
        for i in range(len(self.X)):
                    for j in range(self.X.shape[-1]):
                        if(i==0 and j==0):
                            marginal_contribs[idxs[i], idxf[j]]=temp_contribs[i,j]
                        if(i==0 and j>0):
                            marginal_contribs[idxs[i], idxf[j]]=temp_contribs[i,j]-temp_contribs[i,j-1]
                        if(i>0 and j==0):
                            marginal_contribs[idxs[i], idxf[j]]=temp_contribs[i,j]-temp_contribs[i-1,j]
                        if(i>0 and j>0):
                            marginal_contribs[idxs[i], idxf[j]]=temp_contribs[i,j]+temp_contribs[i-1,j-1]-temp_contribs[i,j-1]-temp_contribs[i-1,j]
        self.margin = marginal_contribs
        
        #print(marginal_contribs)
        return marginal_contribs, idxs
    
    def restart_model(self):
        
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
        
   
    
    def _merge_parallel_results(self, key, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) 
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.X.shape[0]*self.X.shape[-1]))
        n_sources = len(self.X) if self.sources is None else len(self.sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.X)*self.X.shape[-1])
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory, 
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory, 
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self, max_samples=None):
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        if self.model_family not in ['logistic', 'NN']:
            return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    
    