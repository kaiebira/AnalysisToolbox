#!/bin/python3

#
# main_getBeta.py
#
# D. Clarke, K. Ebira, Google Antigravity
#
# Given an Nt and T, get beta.
#

import argparse
from latqcdtools.physics.referenceScales import r0_div_a, a_times_fk, CY_phys, CY_param, ignoreBetaRange, get_aWorld
from latqcdtools.physics.constants import fk_phys, r0_phys
from latqcdtools.base.utilities import getArgs, unvector
from latqcdtools.math.optimize import persistentSolve
import latqcdtools.base.logger as logger

logger.set_log_level('INFO')
ignoreBetaRange()

parser = argparse.ArgumentParser(description='Compute a and T.')

parser.add_argument('--Nt', dest='Nt', type=int, default=8, help='euclidean time extension')
parser.add_argument('--Nf', dest='Nf', type=str, default='21', help='number flavors, e.g. 21')
parser.add_argument('--scale', dest='scale', required=True, help='reference scale (r0, r1, or fk)')
parser.add_argument('--T', dest='T', required=True, type=float, help='temperature in [MeV]')
parser.add_argument('--paramYear', dest='paramYear', type=int, default=None, help='year for a(beta) parameterization')
parser.add_argument('--scaleYear', dest='scaleYear', type=int, default=None, help='year for scale in physical units')

args = getArgs(parser)

scale      = args.scale
T          = args.T
Nt         = args.Nt
Nf         = args.Nf
scaleYear  = args.scaleYear
paramYear  = args.paramYear
beta_guess = 6.0

if Nf is not None and Nf.strip().lower() in ('none', '0'):
    Nf = None

# Implement target_T - T as the LHS of the solver.
if scale == 'r0':
    if paramYear is None:
        paramYear = 2012 if Nf == '21' else CY_param['r0']
    if scaleYear is None:
        scaleYear = CY_phys['r0']
    physWorld = 'Nf21'
    def LHS(beta):
        return r0_div_a(beta,paramYear)/r0_phys(scaleYear,units="MeVinv",world=physWorld)/Nt - T
elif scale == 'fk':
    if paramYear is None:
        paramYear = CY_param['fk']
    if scaleYear is None:
        scaleYear = CY_phys['fk']
    physWorld = f'Nf{Nf}' if (Nf is not None and f'Nf{Nf}' in ('Nf211', 'Nf21', 'Nf2')) else 'Nf21'
    def LHS(beta):
        return fk_phys(scaleYear,units="MeV",world=physWorld)/a_times_fk(beta,paramYear)/Nt - T
else:
    logger.TBError('Beta extraction not yet implemented for scale',scale)

ensemble_world = 'Nf0' if Nf is None else f'Nf{Nf}'
aWorld = get_aWorld(scale, paramYear)
if ensemble_world != aWorld:
    logger.warn(f"Ensemble world '{ensemble_world}' does not match parameterization world '{aWorld}'.")

logger.info('     scale =',scale)
logger.info('        Nf =',Nf)
logger.info(' scaleYear =',scaleYear)
logger.info(' paramYear =',paramYear)
logger.info('         T =',T,'[MeV]')
logger.info('        Nt =',Nt)

beta_solution = persistentSolve(LHS,beta_guess)

logger.info('  beta_sol =',round(unvector(beta_solution),3))
