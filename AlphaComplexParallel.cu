/*********************************************************************************
 *
 * Copyright (c) 2019 Visualization & Graphics Lab (VGL), Indian Institute of Science
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Author   : Talha Bin Masood
 * Contact  : talha [AT] iisc.ac.in
 * Citation : T. B. Masood, T. Ray and V. Natarajan. 
 *            "Parallel Computation of Alpha Complex for Biomolecules"
 *            https://arxiv.org/abs/1908.05944
 *********************************************************************************/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/merge.h>
#include <thrust/iterator/constant_iterator.h>
#include "timer.h"
#include "predicates.h"
#include <iostream>

#define THRES (0.0)
#define ATHRES (0.0)

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }

inline void gpuAssert(cudaError_t code, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %d\n", cudaGetErrorString(code), line);
	}
}

typedef struct At {
	float x, y, z;
	float radius;
	unsigned int index;
} Atom;

__global__ void computeCellIndex(Atom *atomList, unsigned int *indexList,
		unsigned int numAtoms, float3 min, float3 max, int3 gridSize,
		float stepSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numAtoms)
		return;
	Atom atom = atomList[i];
	int cellX = (int) ((atom.x - min.x) / stepSize);
	int cellY = (int) ((atom.y - min.y) / stepSize);
	int cellZ = (int) ((atom.z - min.z) / stepSize);
	int cellIndex = cellX + gridSize.x * cellY
			+ gridSize.x * gridSize.y * cellZ;
	indexList[i] = cellIndex;
}

__global__ void computePossibleEdgeCount(unsigned int *beginList,
		unsigned int *endList, unsigned int *cellIndexList,
		unsigned int numAtoms, unsigned int *edgeCount, int3 gridSize) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	int cellIndex = cellIndexList[atomIndex];
	int temp = cellIndex;
	int gridX = temp % gridSize.x;
	temp = temp / gridSize.x;
	int gridY = temp % gridSize.y;
	int gridZ = temp / gridSize.y;
	unsigned int totalEdgeCount = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				if ((gridZ + i >= gridSize.z) || (gridY + j >= gridSize.y)
						|| (gridX + k >= gridSize.x))
					continue;
				if ((gridZ + i < 0) || (gridY + j < 0) || (gridX + k < 0))
					continue;
				int adjacentCellIndex = (gridZ + i) * gridSize.x * gridSize.y
						+ (gridY + j) * gridSize.x + (gridX + k);
				if (adjacentCellIndex < cellIndex)
					continue;
				unsigned int beginIndex = beginList[adjacentCellIndex];
				unsigned int endIndex = endList[adjacentCellIndex];
				for (int l = beginIndex; l < endIndex; l++) {
					if (l > atomIndex) {
						totalEdgeCount++;
					}
				}
			}
		}
	}
	edgeCount[atomIndex] = totalEdgeCount;
}

__global__ void fillPotentialEdges(unsigned int *beginList,
		unsigned int *endList, unsigned int *cellIndexList,
		unsigned int numAtoms, unsigned int *edgeAddressList,
		unsigned int *edgeLeftList, unsigned int *edgeRightList,
		int3 gridSize) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	int cellIndex = cellIndexList[atomIndex];
	int temp = cellIndex;
	int gridX = temp % gridSize.x;
	temp = temp / gridSize.x;
	int gridY = temp % gridSize.y;
	int gridZ = temp / gridSize.y;
	unsigned int startAddress =
			atomIndex == 0 ? 0 : edgeAddressList[atomIndex - 1];
	unsigned int totalEdgeCount = 0;
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				if ((gridZ + i >= gridSize.z) || (gridY + j >= gridSize.y)
						|| (gridX + k >= gridSize.x))
					continue;
				if ((gridZ + i < 0) || (gridY + j < 0) || (gridX + k < 0))
					continue;
				int adjacentCellIndex = (gridZ + i) * gridSize.x * gridSize.y
						+ (gridY + j) * gridSize.x + (gridX + k);
				if (adjacentCellIndex < cellIndex)
					continue;
				unsigned int beginIndex = beginList[adjacentCellIndex];
				unsigned int endIndex = endList[adjacentCellIndex];
				for (int l = beginIndex; l < endIndex; l++) {
					if (l > atomIndex) {
						edgeLeftList[startAddress + totalEdgeCount] = atomIndex;
						edgeRightList[startAddress + totalEdgeCount] = l;
						totalEdgeCount++;
					}
				}
			}
		}
	}
}

__global__ void markRealEdges(unsigned int *edgeLeftList,
		unsigned int *edgeRightList, bool *edgeMarkList, Atom *atomList,
		int numAtoms, int numEdges, float alpha) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numEdges)
		return;
	Atom a1 = atomList[edgeLeftList[i]];
	Atom a2 = atomList[edgeRightList[i]];
	float a = a1.x - a2.x;
	float b = a1.y - a2.y;
	float c = a1.z - a2.z;
	float d = (a1.x * a1.x - a2.x * a2.x) + (a1.y * a1.y - a2.y * a2.y)
			+ (a1.z * a1.z - a2.z * a2.z) - (a1.radius * a1.radius)
			+ (a2.radius * a2.radius);
	d /= 2;
	float t = (d - a * a1.x - b * a1.y - c * a1.z);
	float currentDist = ((t * t) / (a * a + b * b + c * c))
			- (a1.radius * a1.radius);
	if (currentDist <= (alpha + ATHRES)) {
		edgeMarkList[i] = true;
	} else {
		edgeMarkList[i] = false;
	}
}

__global__ void computePossibleTriangleCount(unsigned int *edgeLeftIndexList,
		unsigned int *edgeRightIndexList, unsigned int *beginIndexList,
		unsigned int *endIndexList, unsigned int *triCountList, int numAtoms) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	unsigned int beginIndex = beginIndexList[atomIndex];
	unsigned int endIndex = endIndexList[atomIndex];
	// Check if there are only one or no edges incident on this atom
	unsigned int numIncidentEdges = endIndex - beginIndex;
	if (numIncidentEdges < 2)
		return;
	unsigned int triCount = 0;
	for (int i = beginIndex; i < endIndex; i++) {
		unsigned int v1 = edgeRightIndexList[i];
		unsigned int v1beg = beginIndexList[v1];
		unsigned int v1end = endIndexList[v1];
		for (int j = i + 1; j < endIndex; j++) {
			unsigned int v2 = edgeRightIndexList[j];
			unsigned int temp;
			if (v2 < v1) {
				temp = v1;
				v1 = v2;
				v2 = temp;
			}
			for (int k = v1beg; k < v1end; k++) {
				if (edgeRightIndexList[k] == v2) {
					triCount++;
					break;
				}
			}
		}
	}
	triCountList[atomIndex] = triCount;
}

__global__ void fillPossibleTriangles(unsigned int *edgeLeftIndexList,
		unsigned int *edgeRightIndexList, unsigned int *beginIndexList,
		unsigned int *endIndexList, int numAtoms, unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList,
		unsigned int *startAddressList) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	unsigned int beginIndex = beginIndexList[atomIndex];
	unsigned int endIndex = endIndexList[atomIndex];
	// Check if there are only one or no edges incident on this atom
	unsigned int numIncidentEdges = endIndex - beginIndex;
	if (numIncidentEdges < 2)
		return;
	unsigned int startAddress =
			atomIndex == 0 ? 0 : startAddressList[atomIndex - 1];
	unsigned int triCount = 0;
	for (int i = beginIndex; i < endIndex; i++) {
		unsigned int v1 = edgeRightIndexList[i];
		for (int j = i + 1; j < endIndex; j++) {
			unsigned int v2 = edgeRightIndexList[j];
			unsigned int temp;
			if (v2 < v1) {
				temp = v1;
				v1 = v2;
				v2 = temp;
			}
			unsigned int v1beg = beginIndexList[v1];
			unsigned int v1end = endIndexList[v1];
			for (int k = v1beg; k < v1end; k++) {
				if (edgeRightIndexList[k] == v2) {
					triV1IndexList[startAddress + triCount] = atomIndex;
					triV2IndexList[startAddress + triCount] = v1;
					triV3IndexList[startAddress + triCount] = v2;
					triCount++;
					break;
				}
			}
		}
	}
}

__global__ void markRealTriangles(unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList,
		bool *triMarkList, float4 *orthoSphereList, Atom *atomList,
		int numAtoms, int numTris, float alpha) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTris)
		return;
	int v1 = triV1IndexList[i];
	int v2 = triV2IndexList[i];
	int v3 = triV3IndexList[i];
	Atom a1 = atomList[v1];
	Atom a2 = atomList[v2];
	Atom a3 = atomList[v3];
	// Find equation of plane of intersection of atoms a1 and a2
	float a11 = a1.x - a2.x;
	float a12 = a1.y - a2.y;
	float a13 = a1.z - a2.z;
	float b1 = (a1.x * a1.x - a2.x * a2.x) + (a1.y * a1.y - a2.y * a2.y)
			+ (a1.z * a1.z - a2.z * a2.z) - (a1.radius * a1.radius)
			+ (a2.radius * a2.radius);
	b1 /= 2;
	// Find equation of plane of intersection of atoms a2 and a3
	float a21 = a2.x - a3.x;
	float a22 = a2.y - a3.y;
	float a23 = a2.z - a3.z;
	float b2 = (a2.x * a2.x - a3.x * a3.x) + (a2.y * a2.y - a3.y * a3.y)
			+ (a2.z * a2.z - a3.z * a3.z) - (a2.radius * a2.radius)
			+ (a3.radius * a3.radius);
	b2 /= 2;
	// Find equation of plane containing centers of atoms a1, a2 and a3
	float a31 = (a2.y - a1.y) * (a3.z - a1.z) - (a3.y - a1.y) * (a2.z - a1.z);
	float a32 = (a2.z - a1.z) * (a3.x - a1.x) - (a3.z - a1.z) * (a2.x - a1.x);
	float a33 = (a2.x - a1.x) * (a3.y - a1.y) - (a3.x - a1.x) * (a2.y - a1.y);
	float b3 = a31 * a1.x + a32 * a1.y + a33 * a1.z;
	// Use Cramer's rule to find intersection point of these three planes
	float D = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
			- a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
	float Dx = b1 * a22 * a33 + a12 * a23 * b3 + a13 * b2 * a32 - a13 * a22 * b3
			- a12 * b2 * a33 - b1 * a23 * a32;
	float Dy = a11 * b2 * a33 + b1 * a23 * a31 + a13 * a21 * b3 - a13 * b2 * a31
			- b1 * a21 * a33 - a11 * a23 * b3;
	float Dz = a11 * a22 * b3 + a12 * b2 * a31 + b1 * a21 * a32 - b1 * a22 * a31
			- a12 * a21 * b3 - a11 * b2 * a32;
	float X = Dx / D;
	float Y = Dy / D;
	float Z = Dz / D;
	float dist = (a1.x - X) * (a1.x - X) + (a1.y - Y) * (a1.y - Y)
			+ (a1.z - Z) * (a1.z - Z) - a1.radius * a1.radius;
	orthoSphereList[i] = make_float4(X, Y, Z, dist);
	triMarkList[i] = (dist <= (alpha + ATHRES));
}

__global__ void computePossibleTetCount(unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList,
		unsigned int *beginIndexList, unsigned int *endIndexList,
		unsigned int *tetCountList, int numAtoms) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	unsigned int beginIndex = beginIndexList[atomIndex];
	unsigned int endIndex = endIndexList[atomIndex];
	// Check if there are less than 3 tris incident on this atom
	unsigned int numIncidentTris = endIndex - beginIndex;
	if (numIncidentTris < 2)
		return;
	unsigned int tetCount = 0;
	int v2Index = beginIndex;
	while (true) {
		if (v2Index >= endIndex)
			break;
		unsigned int v2 = triV2IndexList[v2Index];
		int v2Count = 1;
		while (true) {
			unsigned int currV2Index = v2Index + v2Count;
			if (currV2Index >= endIndex)
				break;
			unsigned int currV2 = triV2IndexList[currV2Index];
			if (currV2 != v2) {
				break;
			}
			v2Count++;
		}
		unsigned int v2beg = beginIndexList[v2];
		unsigned int v2end = endIndexList[v2];
		for (int j = v2Index; j < v2Index + v2Count; j++) {
			unsigned int v3 = triV3IndexList[j];
			for (int k = j + 1; k < v2Index + v2Count; k++) {
				unsigned int v4 = triV3IndexList[k];
				bool v1v3v4Found = false;
				for (int l = k + 1; l < endIndex; l++) {
					if (triV2IndexList[l] == v3 && triV3IndexList[l] == v4) {
						v1v3v4Found = true;
						break;
					}
				}
				if (v1v3v4Found) {
					// check for v2v3v4
					for (int l = v2beg; l < v2end; l++) {
						if (triV2IndexList[l] == v3
								&& triV3IndexList[l] == v4) {
							tetCount++;
							break;
						}
					}
				}
			}
		}
		v2Index += v2Count;
	}
	tetCountList[atomIndex] = tetCount;
}

__global__ void fillPossibleTets(unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList,
		unsigned int *beginIndexList, unsigned int *endIndexList, int numAtoms,
		unsigned int *tetV1IndexList, unsigned int *tetV2IndexList,
		unsigned int *tetV3IndexList, unsigned int *tetV4IndexList,
		unsigned int *startAddressList) {
	int atomIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (atomIndex >= numAtoms)
		return;
	unsigned int beginIndex = beginIndexList[atomIndex];
	unsigned int endIndex = endIndexList[atomIndex];
	// Check if there are less than 3 tris incident on this atom
	unsigned int numIncidentTris = endIndex - beginIndex;
	if (numIncidentTris < 2)
		return;
	unsigned int startAddress =
			atomIndex == 0 ? 0 : startAddressList[atomIndex - 1];
	unsigned int tetCount = 0;
	int v2Index = beginIndex;
	while (true) {
		if (v2Index >= endIndex)
			break;
		unsigned int v2 = triV2IndexList[v2Index];
		int v2Count = 1;
		while (true) {
			unsigned int currV2Index = v2Index + v2Count;
			if (currV2Index >= endIndex)
				break;
			unsigned int currV2 = triV2IndexList[currV2Index];
			if (currV2 != v2) {
				break;
			}
			v2Count++;
		}
		unsigned int v2beg = beginIndexList[v2];
		unsigned int v2end = endIndexList[v2];
		for (int j = v2Index; j < v2Index + v2Count; j++) {
			unsigned int v3 = triV3IndexList[j];
			for (int k = j + 1; k < v2Index + v2Count; k++) {
				unsigned int v4 = triV3IndexList[k];
				bool v1v3v4Found = false;
				for (int l = k + 1; l < endIndex; l++) {
					if (triV2IndexList[l] == v3 && triV3IndexList[l] == v4) {
						v1v3v4Found = true;
						break;
					}
				}
				if (v1v3v4Found) {
					// check for v2v3v4
					for (int l = v2beg; l < v2end; l++) {
						if (triV2IndexList[l] == v3
								&& triV3IndexList[l] == v4) {
							tetV1IndexList[startAddress + tetCount] = atomIndex;
							tetV2IndexList[startAddress + tetCount] = v2;
							tetV3IndexList[startAddress + tetCount] = v3;
							tetV4IndexList[startAddress + tetCount] = v4;
							tetCount++;
							break;
						}
					}
				}
			}
		}
		v2Index += v2Count;
	}
}

__global__ void markRealTets(unsigned int *tetV1IndexList,
		unsigned int *tetV2IndexList, unsigned int *tetV3IndexList,
		unsigned int *tetV4IndexList, bool *tetMarkList, Atom *atomList,
		float4 *orthoSphereList, int numAtoms, int numTets, float alpha) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTets)
		return;
	unsigned int v1 = tetV1IndexList[i];
	unsigned int v2 = tetV2IndexList[i];
	unsigned int v3 = tetV3IndexList[i];
	unsigned int v4 = tetV4IndexList[i];
	Atom a1 = atomList[v1];
	Atom a2 = atomList[v2];
	Atom a3 = atomList[v3];
	Atom a4 = atomList[v4];
	// Find equation of plane of intersection of atoms a1 and a2
	float a11 = a1.x - a2.x;
	float a12 = a1.y - a2.y;
	float a13 = a1.z - a2.z;
	float b1 = (a1.x * a1.x - a2.x * a2.x) + (a1.y * a1.y - a2.y * a2.y)
			+ (a1.z * a1.z - a2.z * a2.z) - (a1.radius * a1.radius)
			+ (a2.radius * a2.radius);
	b1 /= 2;
	// Find equation of plane of intersection of atoms a2 and a3
	float a21 = a2.x - a3.x;
	float a22 = a2.y - a3.y;
	float a23 = a2.z - a3.z;
	float b2 = (a2.x * a2.x - a3.x * a3.x) + (a2.y * a2.y - a3.y * a3.y)
			+ (a2.z * a2.z - a3.z * a3.z) - (a2.radius * a2.radius)
			+ (a3.radius * a3.radius);
	b2 /= 2;
	// Find equation of plane of intersection of atoms a3 and a4
	float a31 = a3.x - a4.x;
	float a32 = a3.y - a4.y;
	float a33 = a3.z - a4.z;
	float b3 = (a3.x * a3.x - a4.x * a4.x) + (a3.y * a3.y - a4.y * a4.y)
			+ (a3.z * a3.z - a4.z * a4.z) - (a3.radius * a3.radius)
			+ (a4.radius * a4.radius);
	b3 /= 2;
	// Use Cramer's rule to find ortho-sphere center
	float D = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
			- a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32;
	float Dx = b1 * a22 * a33 + a12 * a23 * b3 + a13 * b2 * a32 - a13 * a22 * b3
			- a12 * b2 * a33 - b1 * a23 * a32;
	float Dy = a11 * b2 * a33 + b1 * a23 * a31 + a13 * a21 * b3 - a13 * b2 * a31
			- b1 * a21 * a33 - a11 * a23 * b3;
	float Dz = a11 * a22 * b3 + a12 * b2 * a31 + b1 * a21 * a32 - b1 * a22 * a31
			- a12 * a21 * b3 - a11 * b2 * a32;
	if (D == 0.0) {
		tetMarkList[i] = true;
		orthoSphereList[i] = make_float4(0, 0, 0, 0);
	} else {
		float X = Dx / D;
		float Y = Dy / D;
		float Z = Dz / D;
		float dist = (a2.x - X) * (a2.x - X) + (a2.y - Y) * (a2.y - Y)
				+ (a2.z - Z) * (a2.z - Z) - a2.radius * a2.radius;
		tetMarkList[i] = (dist <= (alpha + ATHRES));
		orthoSphereList[i] = make_float4(X, Y, Z, dist);
	}
}

__global__ void initialize() {
	exactinit();
}

__global__ void checkTetsOrthosphere(unsigned int *tetV1IndexList,
		unsigned int *tetV2IndexList, unsigned int *tetV3IndexList,
		unsigned int *tetV4IndexList, Atom *atomList, float3 min,
		unsigned int *beginList, unsigned int *endList, float stepSize,
		int3 gridSize, float4 *orthoSphereList, int numSpheres,
		bool *sphereMarkList) {
	int tetIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (tetIndex >= numSpheres)
		return;
	float4 sphere = orthoSphereList[tetIndex];
	unsigned int v1 = tetV1IndexList[tetIndex];
	unsigned int v2 = tetV2IndexList[tetIndex];
	unsigned int v3 = tetV3IndexList[tetIndex];
	unsigned int v4 = tetV4IndexList[tetIndex];
	Atom a1 = atomList[v1];
	Atom a2 = atomList[v2];
	Atom a3 = atomList[v3];
	Atom a4 = atomList[v4];
	float pa[3], pb[3], pc[3], pd[3], pe[3];
	float wa, wb, wc, wd, we;
	pa[0] = a1.x, pa[1] = a1.y, pa[2] = a1.z, wa = a1.radius * a1.radius;
	pb[0] = a2.x, pb[1] = a2.y, pb[2] = a2.z, wb = a2.radius * a2.radius;
	pc[0] = a3.x, pc[1] = a3.y, pc[2] = a3.z, wc = a3.radius * a3.radius;
	pd[0] = a4.x, pd[1] = a4.y, pd[2] = a4.z, wd = a4.radius * a4.radius;
	int gridX = (int) ((sphere.x - min.x) / stepSize);
	int gridY = (int) ((sphere.y - min.y) / stepSize);
	int gridZ = (int) ((sphere.z - min.z) / stepSize);
	// Check the atoms in current cell first because there is higher chance that they will violate the orthosphere condition.
	int currCellIndex = gridZ * gridSize.x * gridSize.y + gridY * gridSize.x
			+ gridX;
	int atomCount = endList[currCellIndex] - beginList[currCellIndex];
	float orientation = orient3d(pa, pb, pc, pd);

	if (atomCount > 0) {
		for (int l = 0; l < atomCount; l++) {
			int atomIndex = beginList[currCellIndex] + l;
			if (atomIndex == v1 || atomIndex == v2 || atomIndex == v3
					|| atomIndex == v4)
				continue;
			Atom atom = atomList[atomIndex];
			pe[0] = atom.x, pe[1] = atom.y, pe[2] = atom.z;
			we = atom.radius * atom.radius;
			if (orientation < 0) {
				float result = insphere_w(pa, pb, pc, pd, pe, wa, wb, wc, wd,
						we);
				if (result < 0) {
					sphereMarkList[tetIndex] = false;
					return;
				} else if (result == 0) {
					result = orientation4SoS_w(pa, pb, pc, pd, pe, wa, wb, wc,
							wd, we, a1.index, a2.index, a3.index, a4.index,
							atom.index);
					if (result < 0) {
						sphereMarkList[tetIndex] = false;
						return;
					}
				}
			} else {
				float result = insphere_w(pa, pb, pc, pd, pe, wa, wb, wc, wd,
						we);
				if (result > 0) {
					sphereMarkList[tetIndex] = false;
					return;
				} else if (result == 0) {
					result = orientation4SoS_w(pa, pb, pc, pd, pe, wa, wb, wc,
							wd, we, a1.index, a2.index, a3.index, a4.index,
							atom.index);
					if (result > 0) {
						sphereMarkList[tetIndex] = false;
						return;
					}
				}
			}
		}
	}
	// Then check the neighboring cells for violations.
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				if (i == 0 && j == 0 && k == 0) {
					continue;
				}
				if ((gridZ + i >= gridSize.z) || (gridY + j >= gridSize.y)
						|| (gridX + k >= gridSize.x))
					continue;
				if ((gridZ + i < 0) || (gridY + j < 0) || (gridX + k < 0))
					continue;
				int otherCellIndex = (gridZ + i) * gridSize.x * gridSize.y
						+ (gridY + j) * gridSize.x + (gridX + k);
				int otherCount = endList[otherCellIndex]
						- beginList[otherCellIndex];
				if (otherCount > 0) {
					//tetNeighbourList[tetIndex] += otherCount;
					for (int l = 0; l < otherCount; l++) {
						int atomIndex = beginList[otherCellIndex] + l;
						if (atomIndex == v1 || atomIndex == v2
								|| atomIndex == v3 || atomIndex == v4)
							continue;
						Atom atom = atomList[atomIndex];
						pe[0] = atom.x, pe[1] = atom.y, pe[2] = atom.z;
						we = atom.radius * atom.radius;
						if (orientation < 0) {
							float result = insphere_w(pa, pb, pc, pd, pe, wa,
									wb, wc, wd, we);
							if (result < 0) {
								sphereMarkList[tetIndex] = false;
								return;
							} else if (result == 0) {
								result = orientation4SoS_w(pa, pb, pc, pd, pe,
										wa, wb, wc, wd, we, a1.index, a2.index,
										a3.index, a4.index, atom.index);
								if (result < 0) {
									sphereMarkList[tetIndex] = false;
									return;
								}
							}
						} else {
							float result = insphere_w(pa, pb, pc, pd, pe, wa,
									wb, wc, wd, we);
							if (result > 0) {
								sphereMarkList[tetIndex] = false;
								return;
							} else if (result == 0) {
								result = orientation4SoS_w(pa, pb, pc, pd, pe,
										wa, wb, wc, wd, we, a1.index, a2.index,
										a3.index, a4.index, atom.index);
								if (result > 0) {
									sphereMarkList[tetIndex] = false;
									return;
								}
							}
						}
					}
				}
			}
		}
	}
	sphereMarkList[tetIndex] = true;
}

__global__ void generateTrisInTets(unsigned int *tetV1IndexList,
		unsigned int *tetV2IndexList, unsigned int *tetV3IndexList,
		unsigned int *tetV4IndexList, int numTets, unsigned int *triV1List,
		unsigned int *triV2List, unsigned int *triV3List) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTets)
		return;
	unsigned int v1 = tetV1IndexList[i];
	unsigned int v2 = tetV2IndexList[i];
	unsigned int v3 = tetV3IndexList[i];
	unsigned int v4 = tetV4IndexList[i];
	triV1List[4 * i] = v1;
	triV2List[4 * i] = v2;
	triV3List[4 * i] = v3;
	triV1List[4 * i + 1] = v1;
	triV2List[4 * i + 1] = v2;
	triV3List[4 * i + 1] = v4;
	triV1List[4 * i + 2] = v1;
	triV2List[4 * i + 2] = v3;
	triV3List[4 * i + 2] = v4;
	triV1List[4 * i + 3] = v2;
	triV2List[4 * i + 3] = v3;
	triV3List[4 * i + 3] = v4;
}

__global__ void markTrisForDeletion(unsigned int *triList1V1,
		unsigned int *triList1V2, unsigned int *triList1V3,
		unsigned int list1Size, unsigned int *triList2V1,
		unsigned int *triList2V2, unsigned int *triList2V3,
		unsigned int list2Size, unsigned int *beginIndexList,
		unsigned int *endIndexList, bool *triMarkList) {
	int triIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (triIndex >= list1Size)
		return;
	unsigned int t1v1 = triList1V1[triIndex];
	unsigned int t1v2 = triList1V2[triIndex];
	unsigned int t1v3 = triList1V3[triIndex];
	unsigned int beginIndex = beginIndexList[t1v1];
	unsigned int endIndex = endIndexList[t1v1];
	unsigned int numEntries = endIndex - beginIndex;
	if (numEntries == 0) {
		triMarkList[triIndex] = true;
		return;
	}
	for (int j = beginIndex; j < endIndex; j++) {
		unsigned int t2v1 = triList2V1[j];
		unsigned int t2v2 = triList2V2[j];
		unsigned int t2v3 = triList2V3[j];
		if (t1v1 == t2v1 && t1v2 == t2v2 && t1v3 == t2v3) {
			triMarkList[triIndex] = false;
			return;
		}
	}
	triMarkList[triIndex] = true;
}

__global__ void checkDanglingTris(unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList,
		Atom *atomList, float3 min, unsigned int *beginList,
		unsigned int *endList, float stepSize, int3 gridSize,
		float4 *orthosphereList, unsigned int numTris, bool *triMarkList) {
	int triIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (triIndex >= numTris)
		return;
	unsigned int v1 = triV1IndexList[triIndex];
	unsigned int v2 = triV2IndexList[triIndex];
	unsigned int v3 = triV3IndexList[triIndex];

	float4 p1 = orthosphereList[triIndex];
	float currentDist = p1.w;
	int gridX = (int) ((p1.x - min.x) / stepSize);
	int gridY = (int) ((p1.y - min.y) / stepSize);
	int gridZ = (int) ((p1.z - min.z) / stepSize);
	// Check the atoms in current cell first.
	int currCellIndex = gridZ * gridSize.x * gridSize.y + gridY * gridSize.x
			+ gridX;
	int atomCount = endList[currCellIndex] - beginList[currCellIndex];
	if (atomCount > 0) {
		for (int l = 0; l < atomCount; l++) {
			int atomIndex = beginList[currCellIndex] + l;
			if (atomIndex == v1 || atomIndex == v2 || atomIndex == v3)
				continue;
			Atom atom = atomList[atomIndex];
			float powDist = (atom.x - p1.x) * (atom.x - p1.x)
					+ (atom.y - p1.y) * (atom.y - p1.y)
					+ (atom.z - p1.z) * (atom.z - p1.z)
					- (atom.radius * atom.radius);
			if (powDist + THRES <= currentDist) {
				triMarkList[triIndex] = false;
				return;
			}
		}
	}
	// Then check the neighboring cells for violations.
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				if (i == 0 && j == 0 && k == 0) {
					continue;
				}
				if ((gridZ + i >= gridSize.z) || (gridY + j >= gridSize.y)
						|| (gridX + k >= gridSize.x))
					continue;
				if ((gridZ + i < 0) || (gridY + j < 0) || (gridX + k < 0))
					continue;
				int otherCellIndex = (gridZ + i) * gridSize.x * gridSize.y
						+ (gridY + j) * gridSize.x + (gridX + k);
				int otherCount = endList[otherCellIndex]
						- beginList[otherCellIndex];
				if (otherCount > 0) {
					for (int l = 0; l < otherCount; l++) {
						int atomIndex = beginList[otherCellIndex] + l;
						if (atomIndex == v1 || atomIndex == v2
								|| atomIndex == v3)
							continue;
						Atom atom = atomList[atomIndex];
						float powDist = (atom.x - p1.x) * (atom.x - p1.x)
								+ (atom.y - p1.y) * (atom.y - p1.y)
								+ (atom.z - p1.z) * (atom.z - p1.z)
								- (atom.radius * atom.radius);
						if (powDist + THRES <= currentDist) {
							triMarkList[triIndex] = false;
							return;
						}
					}
				}
			}
		}
	}
	triMarkList[triIndex] = true;
}

__global__ void generateEdgesInTris(unsigned int *triV1IndexList,
		unsigned int *triV2IndexList, unsigned int *triV3IndexList, int numTris,
		unsigned int *edgeLeftList, unsigned int *edgeRightList) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTris)
		return;
	unsigned int v1 = triV1IndexList[i];
	unsigned int v2 = triV2IndexList[i];
	unsigned int v3 = triV3IndexList[i];
	edgeLeftList[3 * i] = v1;
	edgeRightList[3 * i] = v2;
	edgeLeftList[3 * i + 1] = v1;
	edgeRightList[3 * i + 1] = v3;
	edgeLeftList[3 * i + 2] = v2;
	edgeRightList[3 * i + 2] = v3;
}

__global__ void markEdgesForDeletion(unsigned int *edgeList1V1,
		unsigned int *edgeList1V2, unsigned int list1Size,
		unsigned int *edgeList2V1, unsigned int *edgeList2V2,
		unsigned int list2Size, unsigned int *beginIndexList,
		unsigned int *endIndexList, bool *edgeMarkList) {
	int edgeIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (edgeIndex >= list1Size)
		return;
	unsigned int e1v1 = edgeList1V1[edgeIndex];
	unsigned int e1v2 = edgeList1V2[edgeIndex];
	unsigned int beginIndex = beginIndexList[e1v1];
	unsigned int endIndex = endIndexList[e1v1];
	unsigned int numEntries = endIndex - beginIndex;
	if (numEntries == 0) {
		edgeMarkList[edgeIndex] = true;
		return;
	}
	for (int j = beginIndex; j < endIndex; j++) {
		unsigned int e2v1 = edgeList2V1[j];
		unsigned int e2v2 = edgeList2V2[j];
		if (e1v1 == e2v1 && e1v2 == e2v2) {
			edgeMarkList[edgeIndex] = false;
			return;
		}
	}
	edgeMarkList[edgeIndex] = true;
}

__global__ void checkDanglingEdges(unsigned int *edgeV1IndexList,
		unsigned int *edgeV2IndexList, Atom *atomList, float3 min,
		unsigned int *beginList, unsigned int *endList, float stepSize,
		int3 gridSize, unsigned int numEdges, bool *edgeMarkList) {
	int edgeIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (edgeIndex >= numEdges)
		return;
	unsigned int v1 = edgeV1IndexList[edgeIndex];
	unsigned int v2 = edgeV2IndexList[edgeIndex];

	Atom a1 = atomList[v1];
	Atom a2 = atomList[v2];
	float a = a1.x - a2.x;
	float b = a1.y - a2.y;
	float c = a1.z - a2.z;
	float d = (a1.x * a1.x - a2.x * a2.x) + (a1.y * a1.y - a2.y * a2.y)
			+ (a1.z * a1.z - a2.z * a2.z) - (a1.radius * a1.radius)
			+ (a2.radius * a2.radius);
	d /= 2;
	// Find a point p0 on the intersection plane
	float t = (d - a * a1.x - b * a1.y - c * a1.z) / (a * a + b * b + c * c);
	float px = a1.x + a * t;
	float py = a1.y + b * t;
	float pz = a1.z + c * t;
	float currentDist = (a1.x - px) * (a1.x - px) + (a1.y - py) * (a1.y - py)
			+ (a1.z - pz) * (a1.z - pz) - (a1.radius * a1.radius);

	int gridX = (int) ((px - min.x) / stepSize);
	int gridY = (int) ((py - min.y) / stepSize);
	int gridZ = (int) ((pz - min.z) / stepSize);
	// Check the atoms in current cell first.
	int currCellIndex = gridZ * gridSize.x * gridSize.y + gridY * gridSize.x
			+ gridX;
	int atomCount = endList[currCellIndex] - beginList[currCellIndex];
	if (atomCount > 0) {
		for (int l = 0; l < atomCount; l++) {
			int atomIndex = beginList[currCellIndex] + l;
			if (atomIndex == v1 || atomIndex == v2)
				continue;
			Atom atom = atomList[atomIndex];
			float powDist = (atom.x - px) * (atom.x - px)
					+ (atom.y - py) * (atom.y - py)
					+ (atom.z - pz) * (atom.z - pz)
					- (atom.radius * atom.radius);
			if (powDist + THRES <= currentDist) {
				edgeMarkList[edgeIndex] = false;
				return;
			}
		}
	}

	// Then check the neighboring cells for violations.
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				if (i == 0 && j == 0 && k == 0) {
					continue;
				}
				if ((gridZ + i >= gridSize.z) || (gridY + j >= gridSize.y)
						|| (gridX + k >= gridSize.x))
					continue;
				if ((gridZ + i < 0) || (gridY + j < 0) || (gridX + k < 0))
					continue;
				int otherCellIndex = (gridZ + i) * gridSize.x * gridSize.y
						+ (gridY + j) * gridSize.x + (gridX + k);
				int otherCount = endList[otherCellIndex]
						- beginList[otherCellIndex];
				if (otherCount > 0) {
					for (int l = 0; l < otherCount; l++) {
						int atomIndex = beginList[otherCellIndex] + l;
						if (atomIndex == v1 || atomIndex == v2)
							continue;
						Atom atom = atomList[atomIndex];
						float powDist = (atom.x - px) * (atom.x - px)
								+ (atom.y - py) * (atom.y - py)
								+ (atom.z - pz) * (atom.z - pz)
								- (atom.radius * atom.radius);
						if (powDist + THRES <= currentDist) {
							edgeMarkList[edgeIndex] = false;
							return;
						}
					}
				}
			}
		}
	}
	edgeMarkList[edgeIndex] = true;
}

struct removeEdgeIfFalse {
	__host__ __device__
	float operator()(const thrust::tuple<bool, unsigned int, unsigned int>& t) {
		return (thrust::get<0>(t) == false);
	}
};

struct removeTriangleIfFalse {
	__host__ __device__
	float operator()(
			const thrust::tuple<bool, unsigned int, unsigned int, unsigned int,
			float4>& t) {
		return (thrust::get<0>(t) == false);
	}
};

struct removeTetIfFalse {
	__host__ __device__
	float operator()(
			const thrust::tuple<bool, unsigned int, unsigned int, unsigned int,
			unsigned int, float4>& t) {
		return (thrust::get<0>(t) == false);
	}
};

typedef thrust::tuple<unsigned int, unsigned int> UInt2Tuple;
struct TupleCmp {
	__host__ __device__
	bool operator()(const UInt2Tuple& t1, const UInt2Tuple& t2) {
		if (thrust::get<0>(t1) == thrust::get<0>(t2)) {
			return thrust::get<1>(t1) < thrust::get<1>(t2);
		}
		return thrust::get<0>(t1) < thrust::get<0>(t2);
	}
};

int main(int argc, char **argv) {
	if (argc < 5) {
		printf("Usage : parallelac <crd-file> <out-file> <sol-rad> <alpha>\n");
		return 1;
	}

	float minX, minY, minZ, maxX, maxY, maxZ;
	float minRadius, maxRadius;
	thrust::host_vector<Atom> h_atoms;
	thrust::device_vector<Atom> d_atoms;
	thrust::device_vector<unsigned int> d_atomCellIndices;
	unsigned int numAtoms;

	FILE *input;
	float x, y, z, radius;

	input = fopen(argv[1], "r");

	/* Read the no. of atoms and bonds */
	fscanf(input, "%d", &numAtoms);

	/* allocate required data */
	h_atoms.reserve(numAtoms);

	/* Read the Atoms*/
	float solventRadius = atof(argv[3]);
	float alpha = atof(argv[4]);
	for (int i = 0; i < numAtoms; i++) {
		fscanf(input, "%f %f %f %f", &x, &y, &z, &radius);
		Atom atom;
		atom.x = x;
		atom.y = y;
		atom.z = z;
		atom.radius = radius + solventRadius;
		atom.index = i + 1;
		h_atoms.push_back(atom);
		if (i == 0) {
			minX = maxX = x;
			minY = maxY = y;
			minZ = maxZ = z;
			minRadius = maxRadius = radius;
		} else {
			if (x < minX)
				minX = x;
			else if (x > maxX)
				maxX = x;
			if (y < minY)
				minY = y;
			else if (y > maxY)
				maxY = y;
			if (z < minZ)
				minZ = z;
			else if (z > maxZ)
				maxZ = z;
			if (radius < minRadius)
				minRadius = radius;
			else if (radius > maxRadius)
				maxRadius = radius;
		}
	}
	fclose(input);
	printf("Successfully read the file %s ...\n", argv[1]);
	printf(
			"The given file has %d atoms. \n\nStarting alpha complex computation for "
					"alpha = %.3f and solvent radius = %.3f ...\n\n", numAtoms,
			alpha, solventRadius);

	float possibleMaxRadius = sqrtf(maxRadius * maxRadius + alpha);
	float stepSize = 2 * (possibleMaxRadius);
	int gridExtentX = (int) ((maxX - minX) / stepSize) + 1;
	int gridExtentY = (int) ((maxY - minY) / stepSize) + 1;
	int gridExtentZ = (int) ((maxZ - minZ) / stepSize) + 1;

	HostTimer memTimer, gridTimer, edgeAlphaTimer, triAlphaTimer;
	HostTimer tetAlphaTimer, tetOrthoTimer, triOrthoTimer, edgeOrthTimer;
	memTimer.start();
	d_atoms = h_atoms;
	d_atomCellIndices.resize(numAtoms);
	memTimer.stop();

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	initialize <<<1, 1>>>();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gridTimer.start();
	int THREADS = 512;
	dim3 blocks(numAtoms / THREADS + 1, 1, 1);
	dim3 threads(THREADS, 1, 1);
	Atom *d_atomList_ptr = thrust::raw_pointer_cast(&d_atoms[0]);
	unsigned int *d_cellIndexList_ptr = thrust::raw_pointer_cast(
			&d_atomCellIndices[0]);
	computeCellIndex<<<blocks, threads>>>(d_atomList_ptr, d_cellIndexList_ptr,
			numAtoms, make_float3(minX, minY, minZ),
			make_float3(maxX, maxY, maxZ),
			make_int3(gridExtentX, gridExtentY, gridExtentZ), stepSize);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_vector<unsigned int> d_origAtomIndices(numAtoms);
	thrust::sequence(d_origAtomIndices.begin(), d_origAtomIndices.end(), 1, 1);

	thrust::sort_by_key(d_atomCellIndices.begin(), d_atomCellIndices.end(),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_atoms.begin(),
							d_origAtomIndices.begin())));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gridTimer.stop();

	thrust::host_vector<unsigned int> h_origAtomIndices = d_origAtomIndices;
	thrust::host_vector<int> h_atomCellIndices = d_atomCellIndices;
	cudaDeviceSynchronize();

	thrust::host_vector<unsigned int> h_sortedAtomIndices(numAtoms);
	for (int i = 0; i < numAtoms; i++) {
		h_sortedAtomIndices[h_origAtomIndices[i] - 1] = i;
	}
	cudaDeviceSynchronize();

	d_origAtomIndices.clear();
	cudaDeviceSynchronize();

	gridTimer.restart();
	thrust::device_vector<unsigned int> d_indexBegin(
			gridExtentX * gridExtentY * gridExtentZ);
	thrust::device_vector<unsigned int> d_indexEnd(
			gridExtentX * gridExtentY * gridExtentZ);
	unsigned int *d_indexBegin_ptr = thrust::raw_pointer_cast(&d_indexBegin[0]);
	unsigned int *d_indexEnd_ptr = thrust::raw_pointer_cast(&d_indexEnd[0]);
	thrust::counting_iterator<unsigned int> search_begin(0);
	thrust::lower_bound(d_atomCellIndices.begin(), d_atomCellIndices.end(),
			search_begin,
			search_begin + gridExtentX * gridExtentY * gridExtentZ,
			d_indexBegin.begin());
	thrust::upper_bound(d_atomCellIndices.begin(), d_atomCellIndices.end(),
			search_begin,
			search_begin + gridExtentX * gridExtentY * gridExtentZ,
			d_indexEnd.begin());
	gridTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	edgeAlphaTimer.start();
	thrust::device_vector<unsigned int> d_possibleEdgeCount(numAtoms, 0u);
	unsigned int *d_possibleEdgeCount_ptr = thrust::raw_pointer_cast(
			&d_possibleEdgeCount[0]);
	computePossibleEdgeCount<<<blocks, threads>>>(d_indexBegin_ptr,
			d_indexEnd_ptr, d_cellIndexList_ptr, numAtoms,
			d_possibleEdgeCount_ptr,
			make_int3(gridExtentX, gridExtentY, gridExtentZ));

	thrust::inclusive_scan(d_possibleEdgeCount.begin(),
			d_possibleEdgeCount.end(), d_possibleEdgeCount.begin());
	unsigned int numEdges = d_possibleEdgeCount[d_possibleEdgeCount.size() - 1];
	thrust::device_vector<unsigned int> d_edgeLeftIndex(numEdges);
	thrust::device_vector<unsigned int> d_edgeRightIndex(numEdges);
	thrust::device_vector<bool> d_edgeMarkList(numEdges, false);
	unsigned int *d_edgeLeftIndex_ptr = thrust::raw_pointer_cast(
			&d_edgeLeftIndex[0]);
	unsigned int *d_edgeRightIndex_ptr = thrust::raw_pointer_cast(
			&d_edgeRightIndex[0]);
	bool *d_edgeMark_ptr = thrust::raw_pointer_cast(&d_edgeMarkList[0]);
	fillPotentialEdges<<<blocks, threads>>>(d_indexBegin_ptr, d_indexEnd_ptr,
			d_cellIndexList_ptr, numAtoms, d_possibleEdgeCount_ptr,
			d_edgeLeftIndex_ptr, d_edgeRightIndex_ptr,
			make_int3(gridExtentX, gridExtentY, gridExtentZ));
	blocks = dim3(numEdges / THREADS + 1, 1, 1);
	markRealEdges<<<blocks, threads >>>(d_edgeLeftIndex_ptr,
			d_edgeRightIndex_ptr, d_edgeMark_ptr, d_atomList_ptr, numAtoms,
			numEdges, alpha);
	// Make zip_iterator easy to use
	typedef thrust::device_vector<unsigned int>::iterator UIntDIter;
	typedef thrust::device_vector<bool>::iterator BoolDIter;
	typedef thrust::tuple<BoolDIter, UIntDIter, UIntDIter> BoolIntDIterTuple;
	typedef thrust::zip_iterator<BoolIntDIterTuple> ZipDIter;
	ZipDIter newEnd = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.begin(),
							d_edgeLeftIndex.begin(), d_edgeRightIndex.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.end(),
							d_edgeLeftIndex.end(), d_edgeRightIndex.end())),
			removeEdgeIfFalse());
	// Erase the removed elements from the vectors
	BoolIntDIterTuple endTuple = newEnd.get_iterator_tuple();
	d_edgeMarkList.erase(thrust::get < 0 > (endTuple), d_edgeMarkList.end());
	d_edgeLeftIndex.erase(thrust::get < 1 > (endTuple), d_edgeLeftIndex.end());
	d_edgeRightIndex.erase(thrust::get < 2 > (endTuple),
			d_edgeRightIndex.end());

	thrust::sort_by_key(d_edgeLeftIndex.begin(), d_edgeLeftIndex.end(),
			d_edgeRightIndex.begin());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	edgeAlphaTimer.stop();

	thrust::host_vector<unsigned int> h_edgesNotInTrisListV1 = d_edgeLeftIndex;
	thrust::host_vector<unsigned int> h_edgesNotInTrisListV2 = d_edgeRightIndex;
	cudaDeviceSynchronize();

	edgeAlphaTimer.restart();
	thrust::device_vector<unsigned int> d_edge_beginIndex(numAtoms);
	thrust::device_vector<unsigned int> d_edge_endIndex(numAtoms);

	thrust::lower_bound(d_edgeLeftIndex.begin(), d_edgeLeftIndex.end(),
			search_begin, search_begin + numAtoms, d_edge_beginIndex.begin());
	thrust::upper_bound(d_edgeLeftIndex.begin(), d_edgeLeftIndex.end(),
			search_begin, search_begin + numAtoms, d_edge_endIndex.begin());
	edgeAlphaTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::host_vector<unsigned int> h_edge_beginIndex = d_edge_beginIndex;
	thrust::host_vector<unsigned int> h_edge_endIndex = d_edge_endIndex;
	cudaDeviceSynchronize();

	triAlphaTimer.start();
	thrust::device_vector<unsigned int> d_possibleTriCount(numAtoms, 0u);
	unsigned int *d_possibleTriCount_ptr = thrust::raw_pointer_cast(
			&d_possibleTriCount[0]);
	unsigned int *d_edge_beginIndex_ptr = thrust::raw_pointer_cast(
			&d_edge_beginIndex[0]);
	unsigned int *d_edge_endIndex_ptr = thrust::raw_pointer_cast(
			&d_edge_endIndex[0]);
	blocks = dim3(numAtoms / THREADS + 1, 1, 1);
	computePossibleTriangleCount<<<blocks, threads>>>(d_edgeLeftIndex_ptr,
			d_edgeRightIndex_ptr, d_edge_beginIndex_ptr, d_edge_endIndex_ptr,
			d_possibleTriCount_ptr, numAtoms);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::inclusive_scan(d_possibleTriCount.begin(), d_possibleTriCount.end(),
			d_possibleTriCount.begin());
	unsigned int numTris = 0;
	numTris = d_possibleTriCount[d_possibleTriCount.size() - 1];

	thrust::device_vector<unsigned int> d_triV1Index(numTris, 0u);
	thrust::device_vector<unsigned int> d_triV2Index(numTris, 0u);
	thrust::device_vector<unsigned int> d_triV3Index(numTris, 0u);
	thrust::device_vector<bool> d_triMarkList(numTris, false);
	thrust::device_vector<float4> d_p1List(numTris);
	unsigned int *d_triV1Index_ptr = thrust::raw_pointer_cast(&d_triV1Index[0]);
	unsigned int *d_triV2Index_ptr = thrust::raw_pointer_cast(&d_triV2Index[0]);
	unsigned int *d_triV3Index_ptr = thrust::raw_pointer_cast(&d_triV3Index[0]);
	bool *d_triMarkList_ptr = thrust::raw_pointer_cast(&d_triMarkList[0]);
	float4 *d_p1List_ptr = thrust::raw_pointer_cast(&d_p1List[0]);
	fillPossibleTriangles<<<blocks, threads>>>(d_edgeLeftIndex_ptr,
			d_edgeRightIndex_ptr, d_edge_beginIndex_ptr, d_edge_endIndex_ptr,
			numAtoms, d_triV1Index_ptr, d_triV2Index_ptr, d_triV3Index_ptr,
			d_possibleTriCount_ptr);
	blocks = dim3(numTris / THREADS + 1, 1, 1);
	markRealTriangles<<<blocks, threads>>>(d_triV1Index_ptr, d_triV2Index_ptr,
			d_triV3Index_ptr, d_triMarkList_ptr, d_p1List_ptr, d_atomList_ptr,
			numAtoms, numTris, alpha);
	// Make zip_iterator easy to use
	typedef thrust::device_vector<float4>::iterator F4DIter;
	typedef thrust::tuple<BoolDIter, UIntDIter, UIntDIter, UIntDIter, F4DIter> BoolInt3DIterTuple;
	typedef thrust::zip_iterator<BoolInt3DIterTuple> Zip3DIter;
	Zip3DIter newEnd2 = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.begin(),
							d_triV1Index.begin(), d_triV2Index.begin(),
							d_triV3Index.begin(), d_p1List.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.end(), d_triV1Index.end(),
							d_triV2Index.end(), d_triV3Index.end(),
							d_p1List.end())), removeTriangleIfFalse());
	// Erase the removed elements from the vectors
	BoolInt3DIterTuple endTuple2 = newEnd2.get_iterator_tuple();
	d_triMarkList.erase(thrust::get < 0 > (endTuple2), d_triMarkList.end());
	d_triV1Index.erase(thrust::get < 1 > (endTuple2), d_triV1Index.end());
	d_triV2Index.erase(thrust::get < 2 > (endTuple2), d_triV2Index.end());
	d_triV3Index.erase(thrust::get < 3 > (endTuple2), d_triV3Index.end());
	d_p1List.erase(thrust::get < 4 > (endTuple2), d_p1List.end());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::device_vector<unsigned int> d_tri_beginIndex(numAtoms);
	thrust::device_vector<unsigned int> d_tri_endIndex(numAtoms);

	thrust::lower_bound(d_triV1Index.begin(), d_triV1Index.end(), search_begin,
			search_begin + numAtoms, d_tri_beginIndex.begin());
	thrust::upper_bound(d_triV1Index.begin(), d_triV1Index.end(), search_begin,
			search_begin + numAtoms, d_tri_endIndex.begin());
	triAlphaTimer.stop();

	tetAlphaTimer.start();
	thrust::device_vector<unsigned int> d_possibleTetCount(numAtoms, 0u);
	unsigned int *d_possibleTetCount_ptr = thrust::raw_pointer_cast(
			&d_possibleTetCount[0]);
	unsigned int *d_tri_beginIndex_ptr = thrust::raw_pointer_cast(
			&d_tri_beginIndex[0]);
	unsigned int *d_tri_endIndex_ptr = thrust::raw_pointer_cast(
			&d_tri_endIndex[0]);
	blocks = dim3(numAtoms / THREADS + 1, 1, 1);
	computePossibleTetCount<<<blocks, threads>>>(d_triV1Index_ptr,
			d_triV2Index_ptr, d_triV3Index_ptr, d_tri_beginIndex_ptr,
			d_tri_endIndex_ptr, d_possibleTetCount_ptr, numAtoms);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	thrust::inclusive_scan(d_possibleTetCount.begin(), d_possibleTetCount.end(),
			d_possibleTetCount.begin());
	unsigned int numTets = 0;
	numTets = d_possibleTetCount[d_possibleTetCount.size() - 1];

	thrust::device_vector<unsigned int> d_tetV1Index(numTets, 0u);
	thrust::device_vector<unsigned int> d_tetV2Index(numTets, 0u);
	thrust::device_vector<unsigned int> d_tetV3Index(numTets, 0u);
	thrust::device_vector<unsigned int> d_tetV4Index(numTets, 0u);
	thrust::device_vector<bool> d_tetMarkList(numTets, false);
	thrust::device_vector<float4> d_orthoSphereList(numTets);
	unsigned int *d_tetV1Index_ptr = thrust::raw_pointer_cast(&d_tetV1Index[0]);
	unsigned int *d_tetV2Index_ptr = thrust::raw_pointer_cast(&d_tetV2Index[0]);
	unsigned int *d_tetV3Index_ptr = thrust::raw_pointer_cast(&d_tetV3Index[0]);
	unsigned int *d_tetV4Index_ptr = thrust::raw_pointer_cast(&d_tetV4Index[0]);
	bool *d_tetMarkList_ptr = thrust::raw_pointer_cast(&d_tetMarkList[0]);
	float4 *d_orthoSphereList_ptr = thrust::raw_pointer_cast(
			&d_orthoSphereList[0]);

	fillPossibleTets<<<blocks, threads>>>(d_triV1Index_ptr, d_triV2Index_ptr,
			d_triV3Index_ptr, d_tri_beginIndex_ptr, d_tri_endIndex_ptr,
			numAtoms, d_tetV1Index_ptr, d_tetV2Index_ptr, d_tetV3Index_ptr,
			d_tetV4Index_ptr, d_possibleTetCount_ptr);
	blocks = dim3(numTets / THREADS + 1, 1, 1);
	markRealTets<<<blocks, threads >>>(d_tetV1Index_ptr, d_tetV2Index_ptr,
			d_tetV3Index_ptr, d_tetV4Index_ptr, d_tetMarkList_ptr,
			d_atomList_ptr, d_orthoSphereList_ptr, numAtoms, numTets,
			alpha);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// Make zip_iterator easy to use
	typedef thrust::tuple<BoolDIter, UIntDIter, UIntDIter, UIntDIter, UIntDIter,
			F4DIter> BoolInt4DIterTuple;
	typedef thrust::zip_iterator<BoolInt4DIterTuple> Zip4DIter;
	Zip4DIter newEnd3 = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_tetMarkList.begin(),
							d_tetV1Index.begin(), d_tetV2Index.begin(),
							d_tetV3Index.begin(), d_tetV4Index.begin(),
							d_orthoSphereList.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_tetMarkList.end(), d_tetV1Index.end(),
							d_tetV2Index.end(), d_tetV3Index.end(),
							d_tetV4Index.end(), d_orthoSphereList.end())),
			removeTetIfFalse());
	// Erase the removed elements from the vectors
	BoolInt4DIterTuple endTuple3 = newEnd3.get_iterator_tuple();
	d_tetMarkList.erase(thrust::get < 0 > (endTuple3), d_tetMarkList.end());
	d_tetV1Index.erase(thrust::get < 1 > (endTuple3), d_tetV1Index.end());
	d_tetV2Index.erase(thrust::get < 2 > (endTuple3), d_tetV2Index.end());
	d_tetV3Index.erase(thrust::get < 3 > (endTuple3), d_tetV3Index.end());
	d_tetV4Index.erase(thrust::get < 4 > (endTuple3), d_tetV4Index.end());
	d_orthoSphereList.erase(thrust::get < 5 > (endTuple3),
			d_orthoSphereList.end());
	tetAlphaTimer.stop();

	tetOrthoTimer.start();
	int numTetsAfterAlphaFilter = d_tetMarkList.size();
	threads = dim3(THREADS, 1, 1);
	blocks = dim3(d_orthoSphereList.size() / THREADS + 1, 1, 1);
	checkTetsOrthosphere<<<blocks, threads >>>(d_tetV1Index_ptr,
			d_tetV2Index_ptr, d_tetV3Index_ptr, d_tetV4Index_ptr,
			d_atomList_ptr, make_float3(minX, minY, minZ),
			d_indexBegin_ptr, d_indexEnd_ptr, stepSize,
			make_int3(gridExtentX, gridExtentY, gridExtentZ),
			d_orthoSphereList_ptr, d_orthoSphereList.size(), d_tetMarkList_ptr);
	newEnd3 = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_tetMarkList.begin(),
							d_tetV1Index.begin(), d_tetV2Index.begin(),
							d_tetV3Index.begin(), d_tetV4Index.begin(),
							d_orthoSphereList.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_tetMarkList.end(), d_tetV1Index.end(),
							d_tetV2Index.end(), d_tetV3Index.end(),
							d_tetV4Index.end(), d_orthoSphereList.end())),
			removeTetIfFalse());
	// Erase the removed elements from the vectors
	endTuple3 = newEnd3.get_iterator_tuple();
	d_tetMarkList.erase(thrust::get < 0 > (endTuple3), d_tetMarkList.end());
	d_tetV1Index.erase(thrust::get < 1 > (endTuple3), d_tetV1Index.end());
	d_tetV2Index.erase(thrust::get < 2 > (endTuple3), d_tetV2Index.end());
	d_tetV3Index.erase(thrust::get < 3 > (endTuple3), d_tetV3Index.end());
	d_tetV4Index.erase(thrust::get < 4 > (endTuple3), d_tetV4Index.end());
	d_orthoSphereList.erase(thrust::get < 5 > (endTuple3),
			d_orthoSphereList.end());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	threads = dim3(THREADS, 1, 1);
	int finalTetCount = d_tetMarkList.size();
	thrust::device_vector<unsigned int> d_trisInTetsListV1(finalTetCount * 4);
	thrust::device_vector<unsigned int> d_trisInTetsListV2(finalTetCount * 4);
	thrust::device_vector<unsigned int> d_trisInTetsListV3(finalTetCount * 4);
	unsigned int *d_trisInTetsListV1_ptr = thrust::raw_pointer_cast(
			&d_trisInTetsListV1[0]);
	unsigned int *d_trisInTetsListV2_ptr = thrust::raw_pointer_cast(
			&d_trisInTetsListV2[0]);
	unsigned int *d_trisInTetsListV3_ptr = thrust::raw_pointer_cast(
			&d_trisInTetsListV3[0]);
	blocks = dim3(finalTetCount / THREADS + 1, 1, 1);
	generateTrisInTets<<<blocks, threads>>>(d_tetV1Index_ptr, d_tetV2Index_ptr,
			d_tetV3Index_ptr, d_tetV4Index_ptr, finalTetCount,
			d_trisInTetsListV1_ptr, d_trisInTetsListV2_ptr,
			d_trisInTetsListV3_ptr);
	tetOrthoTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	triOrthoTimer.start();
	thrust::sort_by_key(d_trisInTetsListV3.begin(), d_trisInTetsListV3.end(),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.begin(),
							d_trisInTetsListV2.begin())));
	thrust::sort_by_key(d_trisInTetsListV2.begin(), d_trisInTetsListV2.end(),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.begin(),
							d_trisInTetsListV3.begin())));
	thrust::sort_by_key(d_trisInTetsListV1.begin(), d_trisInTetsListV1.end(),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV2.begin(),
							d_trisInTetsListV3.begin())));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	typedef thrust::tuple<UIntDIter, UIntDIter, UIntDIter> UInt3DIterTuple;
	typedef thrust::zip_iterator<UInt3DIterTuple> ZipU3DIter;
	ZipU3DIter newEnd4 = thrust::unique(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.begin(),
							d_trisInTetsListV2.begin(),
							d_trisInTetsListV3.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.end(),
							d_trisInTetsListV2.end(),
							d_trisInTetsListV3.end())));
	UInt3DIterTuple endTuple4 = newEnd4.get_iterator_tuple();
	d_trisInTetsListV1.erase(thrust::get < 0 > (endTuple4),
			d_trisInTetsListV1.end());
	d_trisInTetsListV2.erase(thrust::get < 1 > (endTuple4),
			d_trisInTetsListV2.end());
	d_trisInTetsListV3.erase(thrust::get < 2 > (endTuple4),
			d_trisInTetsListV3.end());
	cudaDeviceSynchronize();
	int trisInTets = d_trisInTetsListV1.size();

	int potentialTris = d_triV1Index.size();
	blocks = dim3(d_triV1Index.size() / THREADS + 1, 1, 1);
	thrust::device_vector<unsigned int> d_triInTets_beginIndex(numAtoms);
	thrust::device_vector<unsigned int> d_triInTets_endIndex(numAtoms);
	thrust::lower_bound(d_trisInTetsListV1.begin(), d_trisInTetsListV1.end(),
			search_begin, search_begin + numAtoms,
			d_triInTets_beginIndex.begin());
	thrust::upper_bound(d_trisInTetsListV1.begin(), d_trisInTetsListV1.end(),
			search_begin, search_begin + numAtoms,
			d_triInTets_endIndex.begin());
	unsigned int *d_triInTets_beginIndex_ptr = thrust::raw_pointer_cast(
			&d_triInTets_beginIndex[0]);
	unsigned int *d_triInTets_endIndex_ptr = thrust::raw_pointer_cast(
			&d_triInTets_endIndex[0]);
	markTrisForDeletion<<<blocks, threads>>>(d_triV1Index_ptr, d_triV2Index_ptr,
			d_triV3Index_ptr, d_triV1Index.size(), d_trisInTetsListV1_ptr,
			d_trisInTetsListV2_ptr, d_trisInTetsListV3_ptr,
			d_trisInTetsListV1.size(), d_triInTets_beginIndex_ptr,
			d_triInTets_endIndex_ptr, d_triMarkList_ptr);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	newEnd2 = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.begin(),
							d_triV1Index.begin(), d_triV2Index.begin(),
							d_triV3Index.begin(), d_p1List.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.end(), d_triV1Index.end(),
							d_triV2Index.end(), d_triV3Index.end(),
							d_p1List.end())), removeTriangleIfFalse());
	// Erase the removed elements from the vectors
	endTuple2 = newEnd2.get_iterator_tuple();
	d_triMarkList.erase(thrust::get < 0 > (endTuple2), d_triMarkList.end());
	d_triV1Index.erase(thrust::get < 1 > (endTuple2), d_triV1Index.end());
	d_triV2Index.erase(thrust::get < 2 > (endTuple2), d_triV2Index.end());
	d_triV3Index.erase(thrust::get < 3 > (endTuple2), d_triV3Index.end());
	d_p1List.erase(thrust::get < 4 > (endTuple2), d_p1List.end());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	int potentialTrisNotInTets = d_triV1Index.size();

	blocks = dim3(d_triV1Index.size() / THREADS + 1, 1, 1);
	checkDanglingTris<<<blocks, threads >>>(d_triV1Index_ptr, d_triV2Index_ptr,
			d_triV3Index_ptr, d_atomList_ptr,
			make_float3(minX, minY, minZ), d_indexBegin_ptr,
			d_indexEnd_ptr, stepSize,
			make_int3(gridExtentX, gridExtentY, gridExtentZ), d_p1List_ptr,
			d_triV1Index.size(), d_triMarkList_ptr);
	newEnd2 = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.begin(),
							d_triV1Index.begin(), d_triV2Index.begin(),
							d_triV3Index.begin(), d_p1List.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triMarkList.end(), d_triV1Index.end(),
							d_triV2Index.end(), d_triV3Index.end(),
							d_p1List.end())), removeTriangleIfFalse());
	// Erase the removed elements from the vectors
	endTuple2 = newEnd2.get_iterator_tuple();
	d_triMarkList.erase(thrust::get < 0 > (endTuple2), d_triMarkList.end());
	d_triV1Index.erase(thrust::get < 1 > (endTuple2), d_triV1Index.end());
	d_triV2Index.erase(thrust::get < 2 > (endTuple2), d_triV2Index.end());
	d_triV3Index.erase(thrust::get < 3 > (endTuple2), d_triV3Index.end());
	d_p1List.erase(thrust::get < 4 > (endTuple2), d_p1List.end());
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	int alphaTrisNotInTets = d_triV1Index.size();
	thrust::device_vector<unsigned int> d_finaltrisV1(
			trisInTets + alphaTrisNotInTets);
	thrust::device_vector<unsigned int> d_finaltrisV2(
			trisInTets + alphaTrisNotInTets);
	thrust::device_vector<unsigned int> d_finaltrisV3(
			trisInTets + alphaTrisNotInTets);
	unsigned int *d_finaltrisV1_ptr = thrust::raw_pointer_cast(
			&d_finaltrisV1[0]);
	unsigned int *d_finaltrisV2_ptr = thrust::raw_pointer_cast(
			&d_finaltrisV2[0]);
	unsigned int *d_finaltrisV3_ptr = thrust::raw_pointer_cast(
			&d_finaltrisV3[0]);
	thrust::merge(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triV1Index.begin(),
							d_triV2Index.begin(), d_triV3Index.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_triV1Index.end(), d_triV2Index.end(),
							d_triV3Index.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.begin(),
							d_trisInTetsListV2.begin(),
							d_trisInTetsListV3.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_trisInTetsListV1.end(),
							d_trisInTetsListV2.end(),
							d_trisInTetsListV3.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_finaltrisV1.begin(),
							d_finaltrisV2.begin(), d_finaltrisV3.begin())));
	triOrthoTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	int finalTriCount = d_finaltrisV1.size();

	edgeOrthTimer.start();
	thrust::device_vector<unsigned int> d_edgesInTrisListV1(finalTriCount * 3);
	thrust::device_vector<unsigned int> d_edgesInTrisListV2(finalTriCount * 3);
	unsigned int *d_edgesInTrisListV1_ptr = thrust::raw_pointer_cast(
			&d_edgesInTrisListV1[0]);
	unsigned int *d_edgesInTrisListV2_ptr = thrust::raw_pointer_cast(
			&d_edgesInTrisListV2[0]);
	blocks = dim3(finalTriCount / THREADS + 1, 1, 1);
	generateEdgesInTris<<<blocks, threads>>>(d_finaltrisV1_ptr,
			d_finaltrisV2_ptr, d_finaltrisV3_ptr, finalTriCount,
			d_edgesInTrisListV1_ptr, d_edgesInTrisListV2_ptr);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	thrust::sort_by_key(d_edgesInTrisListV2.begin(), d_edgesInTrisListV2.end(),
			d_edgesInTrisListV1.begin());
	thrust::sort_by_key(d_edgesInTrisListV1.begin(), d_edgesInTrisListV1.end(),
			d_edgesInTrisListV2.begin());
	edgeOrthTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	thrust::host_vector<unsigned int> h_edgesInTrisListV1AfterSort =
			d_edgesInTrisListV1;
	thrust::host_vector<unsigned int> h_edgesInTrisListV2AfterSort =
			d_edgesInTrisListV2;
	cudaDeviceSynchronize();
	edgeOrthTimer.restart();
	typedef thrust::tuple<UIntDIter, UIntDIter> UInt2DIterTuple;
	typedef thrust::zip_iterator<UInt2DIterTuple> ZipU2DIter;
	ZipU2DIter newEnd6 = thrust::unique(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgesInTrisListV1.begin(),
							d_edgesInTrisListV2.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgesInTrisListV1.end(),
							d_edgesInTrisListV2.end())));
	UInt2DIterTuple endTuple6 = newEnd6.get_iterator_tuple();
	d_edgesInTrisListV1.erase(thrust::get < 0 > (endTuple6),
			d_edgesInTrisListV1.end());
	d_edgesInTrisListV2.erase(thrust::get < 1 > (endTuple6),
			d_edgesInTrisListV2.end());
	cudaDeviceSynchronize();
	int edgesInTris = d_edgesInTrisListV1.size();

	int potentialEdges = d_edgeLeftIndex.size();
	blocks = dim3(potentialEdges / THREADS + 1, 1, 1);
	thrust::device_vector<unsigned int> d_edgesInTris_beginIndex(numAtoms);
	thrust::device_vector<unsigned int> d_edgesInTris_endIndex(numAtoms);
	thrust::lower_bound(d_edgesInTrisListV1.begin(), d_edgesInTrisListV1.end(),
			search_begin, search_begin + numAtoms,
			d_edgesInTris_beginIndex.begin());
	thrust::upper_bound(d_edgesInTrisListV1.begin(), d_edgesInTrisListV1.end(),
			search_begin, search_begin + numAtoms,
			d_edgesInTris_endIndex.begin());
	unsigned int *d_edgesInTris_beginIndex_ptr = thrust::raw_pointer_cast(
			&d_edgesInTris_beginIndex[0]);
	unsigned int *d_edgesInTris_endIndex_ptr = thrust::raw_pointer_cast(
			&d_edgesInTris_endIndex[0]);

	markEdgesForDeletion<<<blocks, threads >>>(d_edgeLeftIndex_ptr,
			d_edgeRightIndex_ptr, d_edgeLeftIndex.size(),
			d_edgesInTrisListV1_ptr, d_edgesInTrisListV2_ptr,
			d_edgesInTrisListV1.size(), d_edgesInTris_beginIndex_ptr,
			d_edgesInTris_endIndex_ptr, d_edgeMark_ptr);
	edgeOrthTimer.stop();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	h_edgesNotInTrisListV1 = d_edgeLeftIndex;
	h_edgesNotInTrisListV2 = d_edgeRightIndex;
	thrust::host_vector<bool> h_edgeMarkList = d_edgeMarkList;
	cudaDeviceSynchronize();
	edgeOrthTimer.restart();
	newEnd = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.begin(),
							d_edgeLeftIndex.begin(), d_edgeRightIndex.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.end(),
							d_edgeLeftIndex.end(), d_edgeRightIndex.end())),
			removeEdgeIfFalse());

	// Erase the removed elements from the vectors
	endTuple = newEnd.get_iterator_tuple();
	d_edgeMarkList.erase(thrust::get < 0 > (endTuple), d_edgeMarkList.end());
	d_edgeLeftIndex.erase(thrust::get < 1 > (endTuple), d_edgeLeftIndex.end());
	d_edgeRightIndex.erase(thrust::get < 2 > (endTuple),
			d_edgeRightIndex.end());
	cudaDeviceSynchronize();
	edgeOrthTimer.stop();
	int potentialEdgesNotInTris = d_edgeLeftIndex.size();
	h_edgesNotInTrisListV1 = d_edgeLeftIndex;
	h_edgesNotInTrisListV2 = d_edgeRightIndex;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	edgeOrthTimer.restart();
	blocks = dim3(d_edgeLeftIndex.size() / THREADS + 1, 1, 1);
	checkDanglingEdges<<<blocks, threads >>>(d_edgeLeftIndex_ptr,
			d_edgeRightIndex_ptr, d_atomList_ptr,
			make_float3(minX, minY, minZ), d_indexBegin_ptr,
			d_indexEnd_ptr, stepSize,
			make_int3(gridExtentX, gridExtentY, gridExtentZ),
			d_edgeLeftIndex.size(), d_edgeMark_ptr);
	newEnd = thrust::remove_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.begin(),
							d_edgeLeftIndex.begin(), d_edgeRightIndex.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeMarkList.end(),
							d_edgeLeftIndex.end(), d_edgeRightIndex.end())),
			removeEdgeIfFalse());

	// Erase the removed elements from the vectors
	endTuple = newEnd.get_iterator_tuple();
	d_edgeMarkList.erase(thrust::get < 0 > (endTuple), d_edgeMarkList.end());
	d_edgeLeftIndex.erase(thrust::get < 1 > (endTuple), d_edgeLeftIndex.end());
	d_edgeRightIndex.erase(thrust::get < 2 > (endTuple),
			d_edgeRightIndex.end());
	cudaDeviceSynchronize();
	edgeOrthTimer.stop();

	cudaDeviceSynchronize();
	h_edgesNotInTrisListV1 = d_edgeLeftIndex;
	h_edgesNotInTrisListV2 = d_edgeRightIndex;
	cudaDeviceSynchronize();
	edgeOrthTimer.restart();

	int alphaEdgesNotInTris = d_edgeLeftIndex.size();
	thrust::device_vector<unsigned int> d_finalEdgesV1(
			edgesInTris + alphaEdgesNotInTris);
	thrust::device_vector<unsigned int> d_finalEdgesV2(
			edgesInTris + alphaEdgesNotInTris);
	unsigned int *d_finalEdgesV1_ptr = thrust::raw_pointer_cast(
			&d_finalEdgesV1[0]);
	unsigned int *d_finalEdgesV2_ptr = thrust::raw_pointer_cast(
			&d_finalEdgesV2[0]);
	thrust::merge(
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeLeftIndex.begin(),
							d_edgeRightIndex.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgeLeftIndex.end(),
							d_edgeRightIndex.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgesInTrisListV1.begin(),
							d_edgesInTrisListV2.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_edgesInTrisListV1.end(),
							d_edgesInTrisListV2.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(d_finalEdgesV1.begin(),
							d_finalEdgesV2.begin())));
	cudaDeviceSynchronize();
	int finalEdgeCount = d_finalEdgesV1.size();
	edgeOrthTimer.stop();

	memTimer.restart();
	thrust::host_vector<unsigned int> h_tetV1Index = d_tetV1Index;
	thrust::host_vector<unsigned int> h_tetV2Index = d_tetV2Index;
	thrust::host_vector<unsigned int> h_tetV3Index = d_tetV3Index;
	thrust::host_vector<unsigned int> h_tetV4Index = d_tetV4Index;
	cudaDeviceSynchronize();

	thrust::host_vector<unsigned int> h_finaltrisV1 = d_finaltrisV1;
	thrust::host_vector<unsigned int> h_finaltrisV2 = d_finaltrisV2;
	thrust::host_vector<unsigned int> h_finaltrisV3 = d_finaltrisV3;
	cudaDeviceSynchronize();

	thrust::host_vector<unsigned int> h_finalEdgesV1 = d_finalEdgesV1;
	thrust::host_vector<unsigned int> h_finalEdgesV2 = d_finalEdgesV2;
	memTimer.stop();

	printf("Alpha complex computed.\n\n");
	printf("Number of vertices: %15d\n", numAtoms);
	printf("Number of edges: %18ld\n", h_finalEdgesV1.size());
	printf("Number of triangles: %14ld\n", h_finaltrisV1.size());
	printf("Number of tetrahedra: %13ld\n", h_tetV1Index.size());
	printf("Total simplices: %18ld\n",
			(numAtoms + h_finalEdgesV1.size() + h_finaltrisV1.size()
					+ h_tetV1Index.size()));

	printf(
			"\nTime taken (ms):\nData transfer: %20.3f\nGrid computation: %17.3f\nPotential edges: %18.3f\n"
					"Potential triangles: %14.3f\nPotential tetrahedra: %13.3f\nAC2 test for tetrahedra: %10.3f\n"
					"AC2 test for triangles: %11.3f\nAC2 test for edges: %15.3f\n",
			memTimer.value() * 1000, gridTimer.value() * 1000,
			edgeAlphaTimer.value() * 1000, triAlphaTimer.value() * 1000,
			tetAlphaTimer.value() * 1000, tetOrthoTimer.value() * 1000,
			triOrthoTimer.value() * 1000, edgeOrthTimer.value() * 1000);
	printf("Total time taken (ms): %12.3f\n",
			(memTimer.value() + gridTimer.value() + edgeAlphaTimer.value()
					+ triAlphaTimer.value() + tetAlphaTimer.value()
					+ tetOrthoTimer.value() + triOrthoTimer.value()
					+ edgeOrthTimer.value()) * 1000);

	printf("\nWriting computed alpha complex to the file %s ...\n", argv[2]);

	FILE *fp = fopen(argv[2], "w");
	fprintf(fp, "%d %ld %ld %ld\n", numAtoms, h_finalEdgesV1.size(),
			h_finaltrisV1.size(), h_tetV1Index.size());

	for (int i = 0; i < numAtoms; i++) {
		fprintf(fp, "%9.5lf %9.5lf %9.5lf %9.5lf\n", h_atoms[i].x, h_atoms[i].y,
				h_atoms[i].z, h_atoms[i].radius);
	}
	for (int i = 0; i < h_finalEdgesV1.size(); i++) {
		fprintf(fp, "%9d %9d\n", h_finalEdgesV1[i], h_finalEdgesV2[i]);
	}
	for (int i = 0; i < h_finaltrisV1.size(); i++) {
		fprintf(fp, "%9d %9d %9d\n", h_finaltrisV1[i], h_finaltrisV2[i],
				h_finaltrisV3[i]);
	}
	for (int i = 0; i < h_tetV1Index.size(); i++) {
		fprintf(fp, "%9d %9d %9d %9d\n", h_tetV1Index[i], h_tetV2Index[i],
				h_tetV3Index[i], h_tetV4Index[i]);
	}

	h_atoms.clear();
	d_atoms.clear();
	d_atomCellIndices.clear();
	d_origAtomIndices.clear();
	d_finaltrisV1.clear();
	d_finaltrisV2.clear();
	d_finaltrisV3.clear();
	d_finalEdgesV1.clear();
	d_finalEdgesV2.clear();
	d_tetV1Index.clear();
	d_tetV2Index.clear();
	d_tetV3Index.clear();
	d_tetV4Index.clear();

	h_finalEdgesV1.clear();
	h_finalEdgesV2.clear();
	h_finaltrisV1.clear();
	h_finaltrisV2.clear();
	h_finaltrisV3.clear();
	h_tetV1Index.clear();
	h_tetV2Index.clear();
	h_tetV3Index.clear();
	h_tetV4Index.clear();

	return 0;
}

