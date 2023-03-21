from scipy import linalg as la
import numpy as np
import scipy as sp
import networkx as nx
import collections
import torch as th


# NOISE:
# simulates statistical errors bases on finite amount of measured copies
# the error can be approximated as Gaussian,
# with sigma = sqrt((1-v^2)/N), where v is the actual value
# e.g. with v = 0 (i.e. 50:50 probability), the error scales with 1/sqrt(N)
# and thus one needs >10^4 copies for <1% error
# approx is good for N >= 8, below that, the Gaussian approximation fails
# and the correct Poissonian distribution would need to be considered
# automatically chooses a (not-necessarily integer) number
# N of copies that yield this error for a value of 0

################################################

# qubox
# [STATES]
# Single Qubit
H = np.array([[1], [0]])
V = np.array([[0], [1]])
P = np.array([[1], [1]]) / np.sqrt(2)
M = np.array([[1], [-1]]) / np.sqrt(2)
R = np.array([[1], [1j]]) / np.sqrt(2)
L = np.array([[1], [-1j]]) / np.sqrt(2)
# Two Qubits
HH = np.array([[1], [0], [0], [0]])
HV = np.array([[0], [1], [0], [0]])
VH = np.array([[0], [0], [1], [0]])
VV = np.array([[0], [0], [0], [1]])
PP = np.array([[1], [1], [1], [1]])/2
PM = np.array([[1], [-1], [1], [-1]])/2
MP = np.array([[1], [1], [-1], [-1]])/2
MM = np.array([[1], [-1], [-1], [1]])/2
RR = np.array([[1], [1j], [1j], [-1]])/2
RL = np.array([[1], [-1j], [1j], [1]])/2
LR = np.array([[1], [1j], [-1j], [1]])/2
LL = np.array([[1], [-1j], [-1j], [-1]])/2
phip = np.array([[1], [0], [0], [1]]) / np.sqrt(2)
phim = np.array([[1], [0], [0], [-1]]) / np.sqrt(2)
psip = np.array([[0], [1], [1], [0]]) / np.sqrt(2)
psim = np.array([[0], [1], [-1], [0]]) / np.sqrt(2)

# [static OPERATORS]
# Pauli
sigma0 = np.array([[1, 0], [0, 1]])
sigmaX = np.array([[0, 1], [1, 0]])
sigmaY = np.array([[0, -1j], [1j, 0]])
sigmaZ = np.array([[1, 0], [0, -1]])
sigma0X = np.kron(sigma0, sigmaX)
sigma0Y = np.kron(sigma0, sigmaY)
sigma0Z = np.kron(sigma0, sigmaZ)
sigmaX0 = np.kron(sigmaX, sigma0)
sigmaY0 = np.kron(sigmaY, sigma0)
sigmaZ0 = np.kron(sigmaZ, sigma0)
sigmaXX = np.kron(sigmaX, sigmaX)
sigmaXY = np.kron(sigmaX, sigmaY)
sigmaXZ = np.kron(sigmaX, sigmaZ)
sigmaYX = np.kron(sigmaY, sigmaX)
sigmaYY = np.kron(sigmaY, sigmaY)
sigmaYZ = np.kron(sigmaY, sigmaZ)
sigmaZX = np.kron(sigmaZ, sigmaX)
sigmaZY = np.kron(sigmaZ, sigmaY)
sigmaZZ = np.kron(sigmaZ, sigmaZ)


def createRndBlVecList(numEl, includeMixed):
    blVecs = np.random.normal(0, 1, [numEl, 3])  # generate random coordinates
    blVecs = blVecs / np.power(np.sum(np.power(blVecs, 2), 1, keepdims=True), 0.5)  # normalize vectors
    if includeMixed:
        blVecs = blVecs * np.power(np.random.uniform(0, 1, [numEl, 1]), 1/3)  # randomize lengths between 1 and 0 (giving more weight to elements at edge)
    return blVecs


def toRhoFromBlVec(blVec):
    blVec = np.array(blVec)
    blVec = np.expand_dims(blVec, 0)
    blVec = np.expand_dims(blVec, [2, 3]) * np.expand_dims(sigma0, [0, 1])
    pauliMats = np.expand_dims(np.array([sigmaX, sigmaY, sigmaZ]), 0)
    pauliPart = np.matmul(blVec, pauliMats)
    rho = (np.expand_dims(sigma0, 0) + np.sum(pauliPart, 1))/2
    rho = np.squeeze(rho)
    return rho


def toRhoFromBlVecList(blVecs):
    numEl = blVecs.shape[0]
    blVecs = np.expand_dims(blVecs, [2, 3]) * np.expand_dims(sigma0, [0, 1])  # add identity matrix to bloch vectors
    pauliMats = np.repeat(np.expand_dims(np.array([sigmaX, sigmaY, sigmaZ]), 0), numEl, 0)  # create pauli matrix array
    pauliPart = np.matmul(blVecs, pauliMats)
    rhos = (np.repeat(np.expand_dims(sigma0, 0), numEl, 0) + np.sum(pauliPart, 1))/2
    return rhos


def createRndRhoList(numEl, includeMixed):
    blVecs = createRndBlVecList(numEl, includeMixed)  # create list of random Bloch vectors
    rndRhos = toRhoFromBlVecList(blVecs)  # convert to density matrices by multiply with pauli basis
    return rndRhos


def qwp(theta):
    return np.array([[np.cos(theta)**2-1j*np.sin(theta)**2, (1+1j)*np.cos(theta)*np.sin(theta)], [(1+1j)*np.cos(theta)*np.sin(theta), -1j*np.cos(theta)**2+np.sin(theta)**2]])


def hwp(theta):
    return np.array([[np.cos(2*theta), np.sin(2*theta)], [np.sin(2*theta), -np.cos(2*theta)]])


def randUnitary(n):
    # see http://arxiv.org/abs/math-ph/0609050 & http://arxiv.org/pdf/math-ph/0609050v2.pdf
    randmat = (np.random.normal(size=(n, n)) + 1j*np.random.normal(size=(n, n)))/np.sqrt(2)  # dice a random normal distributed complex matrix
    Q, R = np.linalg.qr(randmat)  # QR decomposition
    L = np.zeros((n, n))
    di = np.diag_indices(n)
    L[di] = np.real(R[di]/np.abs(R[di]))  # and create a diagonal matrix with 1s and -1s
    un = Q @ L  # this un should be randomly distributed
    return un


def torho(psi):
    if psi.shape[0] == 1:
        return np.conjugate(psi.T) @ psi
    elif psi.shape[1] == 1:
        return psi @ np.conjugate(psi.T)
    else:
        # input is not vector, leave as it is
        return psi


def blVecFromRho(rho):
    blVec = np.zeros((1, 3))
    blVec[0, 0] = np.real(np.trace(rho @ sigmaX))
    blVec[0, 1] = np.real(np.trace(rho @ sigmaY))
    blVec[0, 2] = np.real(np.trace(rho @ sigmaZ))
    return blVec


def adj(mat):
    return np.conj(np.transpose(mat))


def traceout(rhoOrPsi, qubits2tout):
    # IMPORTANT: qubits are named according to zero indexing,
    # so first qubit means 0, etc.
    # convert input to matrix
    rho = torho(rhoOrPsi)
    # determine the number of qubits
    nQB = np.log2(rho.shape[0]).astype(int)
    if np.isscalar(qubits2tout):
        nTrOutQB = 1
    else:
        nTrOutQB = len(qubits2tout)
        # sort the qubits to be traced out
        qubits2tout = np.sort(qubits2tout)
    # convert density matrix to tensor with 2*nQB dimension
    # the new dimensions look like [row1, row2, row3,..., col1, col2, col3]
    tensorRho = np.reshape(rho, 2*np.ones(np.array(2*nQB), int))
    # move parties to be traced out to front
    trOutDims = np.reshape(np.array([qubits2tout, nQB + qubits2tout]).T, 2*nTrOutQB)  # trOutDims is the beginning of the permutation vector, containing indices of rows and columns of qubits to trace out (row and column, qubit after qubit, the transposition ".T" makes sure that rows and columns are grouped together for each qubit)
    permVec = np.concatenate([trOutDims, np.setdiff1d(np.arange(2*nQB), trOutDims)])  # vector with indices of r/c to be traced out, and then all other remaining dimensions
    tensorRho = np.transpose(tensorRho, permVec)  # perform permutation of dimensions (in python a dimension permutation is called a "transposition")
    # trace out by summing whole tensor over diagonal of nTrOut first qubits
    for k in range(nTrOutQB):
        tensorRho = tensorRho[0, 0] + tensorRho[1, 1]  # python automatically inserts trailing slices
        tensorRho = np.squeeze(tensorRho)
    # reshape array to density operator
    return np.reshape(tensorRho, np.array([2**(nQB-nTrOutQB), 2**(nQB-nTrOutQB)]))


def traceoutList(rhoList, qubits2tout):
    # same as traceout but assumes that a list of states is considered, where the first index denotes the position in the list
    # determine the number of qubits
    nQB = np.log2(rhoList.shape[1]).astype(int)
    if np.isscalar(qubits2tout):
        nTrOutQB = 1
    else:
        nTrOutQB = len(qubits2tout)
        # sort the qubits to be traced out
        qubits2tout = np.sort(qubits2tout)
    # convert density matrix list to tensor with 2*nQB+1 dimension (+1 because of the list dimension)
    # the new dimensions look like [list, row1, row2, row3,..., col1, col2, col3]
    shapeNew = 2*np.ones(np.array(2*nQB), int)
    shapeNew = np.insert(shapeNew, 0, rhoList.shape[0])
    tensorRho = np.reshape(rhoList, shapeNew)
    # move parties to be traced out to front
    trOutDims = np.reshape(np.array([qubits2tout, nQB + qubits2tout]).T, 2*nTrOutQB) + 1  # trOutDims is the beginning of the permutation vector, containing indices of rows and columns of qubits to trace out (row and column, qubit after qubit, the transposition ".T" makes sure that rows and columns are grouped together for each qubit)
    permVec = np.concatenate([trOutDims, np.setdiff1d(np.arange(len(shapeNew)), trOutDims)])  # v ector with indices of r/c to be traced out, and then all other remaining dimensions
    tensorRho = np.transpose(tensorRho, permVec)  # perform permutation of dimensions (in python a dimension permutation is called a "transposition")
    # trace out by summing whole tensor over diagonal of nTrOut first qubits
    for k in range(nTrOutQB):
        tensorRho = tensorRho[0, 0] + tensorRho[1, 1]  # python automatically inserts trailing slices
        tensorRho = np.squeeze(tensorRho)
    # reshape array to density operator
    return np.reshape(tensorRho, np.array([rhoList.shape[0], 2**(nQB-nTrOutQB), 2**(nQB-nTrOutQB)]))


def tensor(*varargin):
    noInputs = len(varargin)  # determine the number of inputs
    tensorprod = 1  # The variable tensorprod is used to save the result.
    if noInputs == 2 and (isinstance(varargin[1], int) or varargin[1].size == 1):
        # SPECIAL CASE
        for k in range(varargin[1]):
            tensorprod = np.kron(tensorprod, varargin[0])
    else:
        for k in range(noInputs):
            tensorprod = np.kron(tensorprod, varargin[k])
    return tensorprod


def tensorList(a, b):
    # calculates kronecker product only accross dimensions 1 and 2, while processing dimension 0 element wise
    # works only if a and b have the same amount of elements along dimension 0
    if (len(a.shape) != 3 or len(b.shape) != 3):
        return 1
    if (a.shape[0] != b.shape[0]):
        return 1
    aShOrig = a.shape
    a = np.kron(a, np.ones([1, b.shape[1], b.shape[2]]))
    b = np.kron(np.ones([1, aShOrig[1], aShOrig[2]]), b)
    return a*b


def atleast_4d(x):
    if x.ndim < 4:
        x = np.expand_dims(np.atleast_3d(x), axis=3)
    return x


class genDataChannels:
    def __init__(self, param_dict):

        for key, value in param_dict.items():
            setattr(self, key, value)
        # [PRIVATE]
        self._pnID = self.pID
        self._pnDeph = self.pDeph
        self._pnDephRnd = self.pDephRnd
        self._pnRepl = self.pRepl
        self._pnReplRnd = self.pReplRnd
        self._pnWhNoise = self.pWhNoise
        self._pnMix = self.pMix
        self._pnCNOTsep = self.pCNOTsep
        self._pnCNOTall = self.pCNOTall
        # normalize user given weights to ensure trace preservation
        self._recalcNormalizedWeights()
        self._rotUnPre = [np.eye(2) for _ in range(self.n_nodes)]
        self._rotUnPost = [np.eye(2) for _ in range(self.n_nodes)]
        self._rndUnForDeph = np.eye(2)
        self._rndPureState = H

    def generate(self, adjacency_matrix, numRuns):
        # SETTINGS - Graphical Properties
        n_nodes = self.n_nodes
        topological_order = [i for i in nx.topological_sort(nx.DiGraph(adjacency_matrix))]
        parents_ids = {node: [inx for inx in np.where(adjacency_matrix[:, node] != 0)[0]] for node in range(n_nodes)}
        parents_nbs = {node: len(parent_ids) for node, parent_ids in parents_ids.items()}
        # determine global random elements for this process
        if not self.randomizeEachRunChannel:
            self._rollNewChannelRnds()
        if self.rotPreRandomizeEachRun:
            self._rotUnPre = [None for id_node in range(self.n_nodes)]
            for id_node in range(self.n_nodes):
                self._rotUnPre[id_node] = self._createRotUnitary(
                    self.rotPreStrength[id_node],
                    self.rotPreRandomDir,
                    self.rotPreSigma[id_node],
                )

        if self.rotPostRandomizeEachRun:
            self._rotUnPost = [None for id_node in range(self.n_nodes)]
            for id_node in range(self.n_nodes):
                self._rotUnPost[id_node] = self._createRotUnitary(
                    self.rotPostStrength[id_node],
                    self.rotPostRandomDir,
                    self.rotPostSigma[id_node],
                )

        gen_data_dict = {id_node: np.zeros((numRuns, 3)) for id_node in topological_order}

        for id_node in topological_order:
            if parents_nbs[id_node] == 0:
                blVecs = createRndBlVecList(numRuns, True)
                gen_data_dict[id_node][:, :] = blVecs
            else:
                for k in range(numRuns):
                    parents_blVecs = [gen_data_dict[parent][k, :] for parent in parents_ids[id_node]]
                    gen_data_dict[id_node][k, :] = self._genChildState(parents_blVecs, id_node)
        gen_data = list(collections.OrderedDict(sorted(gen_data_dict.items())).values())

        # add noise to states
        for index, blVecs in enumerate(gen_data):
            gen_data[index] = self._addStatNoise(blVecs, self.zeroErr[index])

        # make states physical
        if self.makePhysical:
            for index, blVecs in enumerate(gen_data):
                # calculate lengths
                factors = np.sqrt(np.sum(blVecs**2, 1))
                # set all factors below 1 to 1 (they do not need to be changed)
                factors[factors <= 1] = 1
                # shorten the unphysical bloch vectors
                gen_data[index] = blVecs / np.expand_dims(factors, 1)

        # merge data from dict to numpy array
        data = np.zeros((numRuns, 3 * len(gen_data)))
        for index, blVecs in enumerate(gen_data):
            data[:, 3*index: 3*index+3] = blVecs
        return th.tensor(data, dtype=th.float32)

    # PRIVATE-CHANNELS:
    def _genChildState(self, parents_blVecs, id_node):
        _pnID = self._pnID[id_node]
        _pnDeph = self._pnDeph[id_node]
        _pnDephRnd = self._pnDephRnd[id_node]
        _pnRepl = self._pnRepl[id_node]
        _pnReplRnd = self._pnReplRnd[id_node]
        _pnWhNoise = self._pnWhNoise[id_node]
        _pnMix = self._pnMix[id_node]
        _pnCNOTsep = self._pnCNOTsep[id_node]
        _pnCNOTall = self._pnCNOTall[id_node]

        if self.randomizeEachRunChannel:
            self._rollNewChannelRnds()
        if self.rotPreRandomizeEachRun:
            _rotUnPre = self._createRotUnitary(
                self.rotPreStrength[id_node],
                self.rotPreRandomDir,
                self.rotPreSigma[id_node],
            )
        else:
            _rotUnPre = self._rotUnPre[id_node]
        if self.rotPostRandomizeEachRun:
            _rotUnPost = self._createRotUnitary(
                self.rotPostStrength[id_node],
                self.rotPostRandomDir,
                self.rotPostSigma[id_node],
            )
        else:
            _rotUnPost = self._rotUnPost[id_node]
        rhos_in = [toRhoFromBlVec(BlVec) for BlVec in parents_blVecs]
        # pre-rotation applied to all parents
        for j, rho in enumerate(rhos_in):
            rhos_in[j] = _rotUnPre @ rho @ adj(_rotUnPre)
        # apply mixture of gates to generate single qubit output state
        rhoCh = _pnMix * self._gtMix(rhos_in) + _pnCNOTsep * self._gtCNOTsep(rhos_in) + _pnCNOTall * self._gtCNOTall(rhos_in)
        # apply channel to child state
        rho_out = _pnID * rhoCh + _pnDeph * self._chDeph(rhoCh) + _pnDephRnd * self._chDephRnd(rhoCh) + _pnRepl * self._chRepl() + _pnReplRnd * self._chReplRnd() + _pnWhNoise * np.eye(2)
        # apply post rotation to child state
        rho_out = _rotUnPost @ rho_out @ adj(_rotUnPost)
        return blVecFromRho(rho_out)

    def _chDeph(self, rho):
        return np.array([[rho[0, 0], 0], [0, rho[1, 1]]])

    def _chDephRnd(self, rho):
        un = self._rndUnForDeph
        rho = un @ rho @ adj(un)
        rho = self._chDeph(rho)
        rho = adj(un) @ rho @ un
        return rho

    def _chRepl(self):
        return np.array([[1, 0], [0, 0]])

    def _chReplRnd(self):
        return self._rndPureState

    def _gtMix(self, rhos_in):
        rho_new = 0
        for rho in rhos_in:
            rho_new = rho_new + rho
        return rho_new / len(rhos_in)

    def _gtCNOTsep(self, rhos_in):
        n = len(rhos_in)
        rho = self._kronMatrixList(rhos_in)  # create joint density matrix of all incoming states
        rho = np.kron(rho, toRhoFromBlVec(np.array([0, 0, 1])))  # add initial state of child
        un = self._createCNOTsep(n+1)  # create gate unitary
        rho_new = un @ rho @ adj(un)  # apply unitary
        return traceout(rho_new, np.array(range(n)))  # traceout all but the child qubit

    def _gtCNOTall(self, rhos_in):
        n = len(rhos_in)
        rho = self._kronMatrixList(rhos_in)  # create joint density matrix of all incoming states
        rho = np.kron(rho, toRhoFromBlVec(np.array([0, 0, 1])))  # add initial state of child
        un = self._createCNOTall(n+1)  # create gate unitary
        rho_new = un @ rho @ adj(un)  # apply unitary
        return traceout(rho_new, np.array(range(n)))  # traceout all but the child qubit

    # PRIVATE-AUX:
    def _rollNewChannelRnds(self):
        self._rndUnForDeph = randUnitary(2)
        self._rndPureState = torho(randUnitary(2) @ H)

    def _recalcNormalizedWeights(self):
        for id_node in range(self.n_nodes):
            # Channel
            weightSumCh = self.pID[id_node] + self.pDeph[id_node] + self.pDephRnd[id_node] + self.pRepl[id_node] + self.pReplRnd[id_node] + self.pWhNoise[id_node]
            self._pnID[id_node] /= weightSumCh
            self._pnDeph[id_node] /= weightSumCh
            self._pnDephRnd[id_node] /= weightSumCh
            self._pnRepl[id_node] /= weightSumCh
            self._pnReplRnd[id_node] /= weightSumCh
            self._pnWhNoise[id_node] /= weightSumCh
            # Gate
            weightSumGt = self.pMix[id_node] + self.pCNOTsep[id_node] + self.pCNOTall[id_node]
            self._pnMix[id_node] /= weightSumGt
            self._pnCNOTsep[id_node] /= weightSumGt
            self._pnCNOTall[id_node] /= weightSumGt

    def _createRotUnitary(self, strength, randomDir, sigma):
        rndUn = randUnitary(2)
        un = la.expm(1j*sigmaY*np.random.normal(loc=strength, scale=sigma)*np.pi/2)
        if randomDir:
            un = adj(rndUn) @ un @ rndUn
        return un

    def _addStatNoise(self, blVecs, zeroErr):
        if zeroErr <= 0 or zeroErr > 1:
            return blVecs
        nC = 1 / zeroErr**2
        sigmas = np.sqrt((1 - blVecs**2) / nC)  # each bloch-vector coordinate has its own sigma based in its value
        return blVecs + np.random.normal(scale=sigmas)

    def _kronMatrixList(self, rhoList):
        rho = 1
        for ind, r in enumerate(rhoList):
            rho = np.kron(rho, r)
        return rho

    def _createCNOTsep(self, n):
        h = 0  # hamiltonian
        pV = toRhoFromBlVec(np.array([0, 0, -1]))
        id2 = np.eye(2)
        for j in range(n-1):
            hC = 1  # hamiltonian component
            for k in range(n-1):
                if k == j:
                    hC = np.kron(hC, pV)
                else:
                    hC = np.kron(hC, id2)
            hC = np.kron(hC, sigmaX)
            h = h+hC
        u = sp.linalg.expm(-1j*np.pi/2*h)
        return np.abs(u)

    def _createCNOTall(self, n):
        p = 1  # projector on all qubits being flipped
        pV = toRhoFromBlVec(np.array([0, 0, -1]))
        for k in range(n-1):
            p = np.kron(p, pV)
        return np.kron(np.eye(2**(n-1))-p, np.eye(2)) + np.kron(p, sigmaX)


def generate_parameters(n_nodes):
    param_dict = {}

    # SETTINGS - GENERAL Channel Properties
    param_dict['n_nodes'] = n_nodes
    param_dict['randomizeEachRunChannel'] = False
    param_dict['rotPreRandomizeEachRun'] = False
    param_dict['rotPostRandomizeEachRun'] = False
    param_dict['rotPreRandomDir'] = True
    param_dict['rotPostRandomDir'] = True
    param_dict['makePhysical'] = True

    # SETTINGS - ROTATIONS
    min_rotPreStrength, max_rotPreStrength = 0, .1
    min_rotPostStrength, max_rotPostStrength = 0, .1
    min_rotPreSigma, max_rotPreSigma = 0, .1
    min_rotPostSigma, max_rotPostSigma = 0, .1
    min_zeroErr, max_zeroErr = 0, .2
    param_dict['rotPreStrength'] = np.random.uniform(min_rotPreStrength, max_rotPreStrength, n_nodes)
    param_dict['rotPostStrength'] = np.random.uniform(min_rotPostStrength, max_rotPostStrength, n_nodes)
    param_dict['rotPreSigma'] = np.random.uniform(min_rotPreSigma, max_rotPreSigma, n_nodes)
    param_dict['rotPostSigma'] = np.random.uniform(min_rotPostSigma, max_rotPostSigma, n_nodes)
    param_dict['zeroErr'] = np.random.uniform(min_zeroErr, max_zeroErr, n_nodes)

    # SETTINGS - CHANNEL
    min_pID, max_pID = 0, .1
    min_pDeph, max_pDeph = 0, 1
    min_pDephRnd, max_pDephRnd = 0, 1
    min_pRepl, max_pRepl = 0, 1
    min_pReplRnd, max_pReplRnd = 0, 1
    min_pWhNoise, max_pWhNoise = 0, 1
    pID = np.random.uniform(min_pID, max_pID, n_nodes)
    pDeph = np.random.uniform(min_pDeph, max_pDeph, n_nodes)
    pDephRnd = np.random.uniform(min_pDephRnd, max_pDephRnd, n_nodes)
    pRepl = np.random.uniform(min_pRepl, max_pRepl, n_nodes)
    pReplRnd = np.random.uniform(min_pReplRnd, max_pReplRnd, n_nodes)
    pWhNoise = np.random.uniform(min_pWhNoise, max_pWhNoise, n_nodes)
    for id_node in range(n_nodes):
        weightSumCh = pID[id_node] + pDeph[id_node] + pDephRnd[id_node] + pRepl[id_node] + pReplRnd[id_node] + pWhNoise[id_node]
        pID[id_node] /= weightSumCh
        pDeph[id_node] /= weightSumCh
        pDephRnd[id_node] /= weightSumCh
        pRepl[id_node] /= weightSumCh
        pReplRnd[id_node] /= weightSumCh
        pWhNoise[id_node] /= weightSumCh
    param_dict['pID'] = pID
    param_dict['pDeph'] = pDeph
    param_dict['pDephRnd'] = pDephRnd
    param_dict['pRepl'] = pRepl
    param_dict['pReplRnd'] = pReplRnd
    param_dict['pWhNoise'] = pWhNoise

    # SETTINGS - GATE
    min_pMix, max_pMix = 0.5, 1
    min_pCNOTsep, max_pCNOTsep = 0, .5
    min_pCNOTall, max_pCNOTall = 0, .5
    pMix = np.random.uniform(min_pMix, max_pMix, n_nodes)
    pCNOTsep = np.random.uniform(min_pCNOTsep, max_pCNOTsep, n_nodes)
    pCNOTall = np.random.uniform(min_pCNOTall, max_pCNOTall, n_nodes)
    for id_node in range(n_nodes):
        weightSumGt = pMix[id_node] + pCNOTsep[id_node] + pCNOTall[id_node]
        pMix[id_node] /= weightSumGt
        pCNOTsep[id_node] /= weightSumGt
        pCNOTall[id_node] /= weightSumGt
    param_dict['pMix'] = pMix
    param_dict['pCNOTsep'] = pCNOTsep
    param_dict['pCNOTall'] = pCNOTall
    return param_dict


def get_random_adjs(scenario, n_samples, predef_adjacencies_path=None):
    if scenario == 'bi_ce':
        adjacency_list = [
            np.array([[0, 1], [0, 0]]),
            np.array([[0, 0], [1, 0]]),
        ]

    elif scenario == 'bi_ceccin':
        adjacency_list = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ]
    elif scenario == 'multi_base':
        adjacency_list = [
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
            np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
            np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]]),
        ]
    elif predef_adjacencies_path is not None:
        adjacencies = np.loadtxt(predef_adjacencies_path, dtype=float)
        n_nodes = int(np.sqrt(adjacencies.shape[1]))
        adjacency_list = [A.reshape(n_nodes, n_nodes).astype(int) for A in adjacencies]

    if len(adjacency_list) == n_samples:
        adjacencies = adjacency_list
    else:
        indecies_A = np.random.choice(range(len(adjacency_list)), size=n_samples)
        adjacencies = [adjacency_list[index_A] for index_A in indecies_A]
    return adjacencies


def get_random_params(scenario, n_samples, n_nodes):
    param_list = []
    if scenario in ['bi_ce', 'bi_ceccin', 'multi_base']:
        for _ in range(n_samples):
            param_dict = generate_parameters(n_nodes=n_nodes)
            param_list.append(param_dict)
    else:
        param_dict = generate_parameters(n_nodes=n_nodes)
        for _ in range(n_samples):
            param_list.append(param_dict)
    return param_list


def channel_generator(adjacencies_list, parameters_list, n_points, saving_path=None):
    n_samples = len(adjacencies_list)
    n_nodes = adjacencies_list[0].shape[0]
    container_d = th.zeros((n_samples, n_points, 3*n_nodes), dtype=th.float32)
    container_a = th.zeros((n_samples, n_nodes, n_nodes), dtype=th.float32)
    for index_sample in range(n_samples):
        adjacency_matrix = adjacencies_list[index_sample]
        param_dict = parameters_list[index_sample]
        generator = genDataChannels(param_dict=param_dict)
        ch_data = generator.generate(adjacency_matrix=adjacency_matrix, numRuns=n_points)
        container_d[index_sample] = ch_data
        container_a[index_sample] = th.tensor(adjacency_matrix, dtype=th.float32)
        if np.mod(index_sample, 10) == 0:
            print(index_sample)
    ch_container = (container_d, container_a)
    if saving_path is not None:
        th.save(ch_container, saving_path)
    return ch_container
