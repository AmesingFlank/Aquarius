/*

#include "SVD.cuh"
__host__ __device__
void computeSVD2(Mat3x3 mat, float3& eVals, float3& v0, float3& v1, float3& v2) {


	float& a11 = mat.r0.x;
	float& a12 = mat.r0.y;
	float& a13 = mat.r0.z;

	float& a21 = mat.r1.x;
	float& a22 = mat.r1.y;
	float& a23 = mat.r1.z;

	float& a31 = mat.r2.x;
	float& a32 = mat.r2.y;
	float& a33 = mat.r2.z;

	float c2 = -a11 - a22 - a33;
	float c1 = a11 * a22 + a11 * a33 + a22 * a33 - a12 * a12 - a13 * a13 - a23 * a23;
	float c0 = a11 * a23 * a23 + a22 * a13 * a13 + a33 * a12 * a12 - a11 * a22 * a33 - 2 * a12 * a13 * a23;

	float p = c2 * c2 - 3 * c1;
	float q = -27.0 * c0 / 2.0 - c2 * c2 * c2 + 9.0 * c1 * c2 / 2.0;

	float temp = abs(27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 27.0 * c0 / 4.0)));
	float phi = atan2(sqrt(temp), q) / 3.0;

	float cosPhi = cos(phi);
	float sinPhi = sin(phi);

	float x1 = 2 * cosPhi;
	float x2 = -cosPhi - sqrt(3) * sinPhi;
	float x3 = -cosPhi + sqrt(3) * sinPhi;

	//eVals.x = x1 * sqrt(p) / 3 - c2 / 3.0;
	//eVals.y = x2 * sqrt(p) / 3 - c2 / 3.0;
	//eVals.z = x3 * sqrt(p) / 3 - c2 / 3.0;

	float3 A1 = { a11,a21,a31 };
	float3 A2 = { a12,a22,a32 };
	float3 e1 = { 1,0,0 };
	float3 e2 = { 0,1,0 };

	//v0 = cross(A1 - eVals.x * e1, A2 - eVals.x * e2);
	//v1 = cross(A1 - eVals.y * e1, A2 - eVals.y * e2);
	//v2 = cross(A1 - eVals.z * e1, A2 - eVals.z * e2);

}

*/