# 
# speedify.py                                                               
# 
# L. Altenkort, D. Clarke 
# 
# Some methods and classes to easily make python code faster. 
#

import os
import numpy as np
import concurrent.futures
from typing import Any, Callable
from latqcdtools.base.check import checkType
import latqcdtools.base.logger as logger
from numba import njit

# Resolve parallelizer dependencies
DEFAULTPARALLELIZER = 'jax'
try:
    import jax
except ModuleNotFoundError:
    DEFAULTPARALLELIZER = 'pathos.pools'
try:
    import pathos.pools
except ModuleNotFoundError:
    DEFAULTPARALLELIZER = 'concurrent.futures'

# Global configuration flags
COMPILE = False
MAXTHREADS = os.cpu_count()
DEFAULTTHREADS = max(1, MAXTHREADS - 2)

def compileON():
    global COMPILE
    COMPILE = True
    logger.info('Using JIT compilation to speed things up.')

def compileOFF():
    global COMPILE
    COMPILE = False
    logger.info('No longer using JIT compilation.')

def handle_jax_array(x: Any) -> Any:
    """Safely convert arrays to JAX arrays"""
    if DEFAULTPARALLELIZER == 'jax' and isinstance(x, (np.ndarray, list, tuple)):
        return jax.numpy.asarray(x)
    return x

def handle_jax_args(args):
    """Convert arguments to JAX arrays where appropriate"""
    return tuple(handle_jax_array(arg) for arg in args)

def handle_jax_kwargs(kwargs):
    """Convert keyword arguments to JAX arrays where appropriate"""
    return {k: handle_jax_array(v) for k, v in kwargs.items()}


def compile(func: Callable) -> Callable:
    """
    Smart compiler that chooses between JAX, numba, or no compilation.
    """
    global COMPILE
    
    if COMPILE and DEFAULTPARALLELIZER == 'jax':
        logger.info(f'JAX-compiling {func.__name__}')
        try:
            @jax.jit
            def jax_func(*args, **kwargs):
                jax_args = handle_jax_args(args)
                jax_kwargs = handle_jax_kwargs(kwargs)
                result = func(*jax_args, **jax_kwargs)
                return result
                
            def wrapped_func(*args, **kwargs):
                try:
                    return jax_func(*args, **kwargs)
                except Exception as e:
                    logger.warn(f"JAX execution failed: {str(e)}. Using original function.")
                    return func(*args, **kwargs)
                    
            return wrapped_func
            
        except Exception as e:
            logger.warn(f'JAX compilation failed: {str(e)}. Falling back to numba.')
            
    if COMPILE:
        logger.info(f'Numba-compiling {func.__name__}')
        return njit(func)
        
    return func

class ComputationClass:
    def __init__(self, function, input_array, args, nproc, parallelizer):
        """ 
        This class contains everything needed to parallelize a function. It allows for you to
        use your favorite parallelization library and pass arguments to the function.

        Args:
            function (func): to-be-parallelized function
            input_array (array-like): run function over this array 
            nproc (int): number of processors 
            parallelizer (str): Which library should I use to parallelize?
            *add_param: Pass any additional parameters as you would to the function
        """
        checkType(int,nproc=nproc)
        checkType("array",input_array=input_array)
        checkType(str,parallelizer=parallelizer)
        self._input_array  = input_array
        self._function     = function
        self._parallelizer = parallelizer
        self._args         = args
        self._nproc        = nproc
        logger.debug('Initializing')
        logger.debug('input_array =',self._input_array)
        logger.debug('args =',self._args)
        logger.debug('parallelizer =',self._parallelizer)
        if nproc > MAXTHREADS:
            logger.warn('We recommend using fewer processes than',MAXTHREADS) 
        self._result = self.parallelization_wrapper() # compute the result when class is initialized

    def __repr__(self) -> str:
        return "ComputationClass"

    def parallelization_wrapper(self):
        if self._nproc==1:
            results = []
            for i in self._input_array:
                results.append(self.pass_argument_wrapper(i)) 
        else:
            if self._parallelizer=='jax':
                jax_input = jax.numpy.asarray(self._input_array)
                jax_args = handle_jax_args(self._args)
                def mapped_func(x):
                    return self._function(x, *jax_args)
                batched = jax.vmap(mapped_func)
                results = batched(jax_input)
                results = np.array(results)
            elif self._parallelizer=='concurrent.futures':
                with concurrent.futures.ProcessPoolExecutor(max_workers=self._nproc) as executor:
                    results=executor.map(self.pass_argument_wrapper, self._input_array)
            elif self._parallelizer=='pathos.pools':
                pool = pathos.pools.ProcessPool(processes=self._nproc)
                results = pool.map(self.pass_argument_wrapper, self._input_array)
                pool.close()
                pool.join() 
                pool.clear()
            else:
                logger.TBError('Unknown parallelizer',self._parallelizer)
            results = list(results)
        return results

    def pass_argument_wrapper(self, single_input):
        return self._function(single_input, *self._args)

    def getResult(self):
        return self._result


def parallel_function_eval(function, input_array, args=(), nproc=DEFAULTTHREADS, parallelizer=DEFAULTPARALLELIZER):
    """ 
    Parallelize a function over an input_array. Effectively this can replace a loop over an array and should
    lead to a performance boost.

    Args:
        function (func): to-be-parallelized function 
        input_array (array-like): array over which it should run 
        nproc (int): number of processes 

    Returns:
        array-like: func(input_array)
    """
    computer = ComputationClass(function, input_array, args=args, nproc=nproc, parallelizer=parallelizer)
    if nproc==1:
        logger.details('Using for-loop instead of',parallelizer)
    return computer.getResult()


def parallel_reduce(function, input_array, args=(), nproc=DEFAULTTHREADS, parallelizer=DEFAULTPARALLELIZER) -> float:
    """ 
    Parallelize a function over an input_array, then sum over the input_array elements. 

    Args:
        function (func): to-be-parallelized function 
        input_array (array-like): array over which it should run 
        nproc (int): number of processes 

    Returns:
        float
    """
    container = parallel_function_eval(function, input_array, nproc=nproc, args=args, parallelizer=parallelizer)
    return np.sum(container)
