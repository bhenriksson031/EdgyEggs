
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/principal_curvature.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/setdiff.h>
#include <igl/grad.h>
#include <igl/parula.h>
#include <igl/eigs.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/harmonic.h>


#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <GA/GA_Attribute.h>
#include <OP/OP_Operator.h>
#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Include.h>
#include <UT/UT_Matrix3.h>
#include <UT/UT_Matrix4.h>
#include <SYS/SYS_Math.h>

#include "converters.hpp"
#include "SOP_IGLDiscreteGeo.hpp"

using namespace SOP_IGL;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;

static PRM_Name names[] = {
    PRM_Name("false_curve_colors", "Add False Curve Colors"),
    PRM_Name("grad_attrib",        "Add Gradient of Attribute (scalar)"),    
    PRM_Name("grad_attrib_name",   "Scalar Attribute Name"),
    PRM_Name("laplacian",          "Laplacian (Smoothing)"),
    PRM_Name("eigenvectors",       "Eigen Decomposition (Disabled)"),
	PRM_Name("area_min",          "Area Minimizer"),
	PRM_Name("fairing",          "Fairing"),
};

static PRM_Range  laplaceRange(PRM_RANGE_PRM, 0, PRM_RANGE_PRM, 10);

PRM_Template
SOP_IGLDiscreteGeometry::myTemplateList[] = {
    PRM_Template(PRM_TOGGLE, 1, &names[0], PRMzeroDefaults),
    PRM_Template(PRM_TOGGLE, 1, &names[1], PRMzeroDefaults),
    PRM_Template(PRM_STRING, 1, &names[2], 0),
    PRM_Template(PRM_INT_J,  1, &names[3], PRMzeroDefaults, 0, &laplaceRange),
    PRM_Template(PRM_INT_J , 1, &names[4], PRMzeroDefaults, 0, &laplaceRange),
	PRM_Template(PRM_TOGGLE, 1, &names[5], 0),
	PRM_Template(PRM_TOGGLE, 1, &names[6], 0),
    PRM_Template(),
};

namespace SOP_IGL {
void compute_curvature(GU_Detail *gdp, MatrixXd &V, MatrixXi &F, const int false_colors=0)
{
    // Alternative discrete mean curvature
    MatrixXd HN;
    Eigen::SparseMatrix<double> L,M,Minv;
    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    // Laplace-Beltrami of position
    HN = -Minv*(L*V);
    // Extract magnitude as mean curvature
    VectorXd H = HN.rowwise().norm();

    // Compute curvature directions via quadric fitting
    MatrixXd PD1,PD2;
    VectorXd PV1,PV2;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
    // mean curvature
    H = 0.5*(PV1+PV2);
   

    GA_RWHandleF      curvature_h(gdp->addFloatTuple(GA_ATTRIB_POINT, "curvature", 1));
    GA_RWHandleV3     tangentu_h(gdp->addFloatTuple(GA_ATTRIB_POINT, "tangentu", 3));
    GA_RWHandleV3     tangentv_h(gdp->addFloatTuple(GA_ATTRIB_POINT, "tangentv", 3));

    GA_Offset ptoff;
    if (curvature_h.isValid() && tangentu_h.isValid() && tangentv_h.isValid()) {
        GA_FOR_ALL_PTOFF(gdp, ptoff) {
            const GA_Index ptidx = gdp->pointIndex(ptoff);
            UT_ASSERT((uint)ptidx < H.rows());
            UT_ASSERT((uint)ptidx < PD1.rows());
            UT_ASSERT((uint)ptidx < PD2.rows());
            const float curv = H((uint)ptidx, 0);
            const UT_Vector3 tnu(PD1((uint)ptidx, 0), PD1((uint)ptidx, 1), PD1((uint)ptidx, 2));
            const UT_Vector3 tnv(PD2((uint)ptidx, 0), PD2((uint)ptidx, 1), PD2((uint)ptidx, 2));
            curvature_h.set(ptoff, curv);
            tangentu_h.set(ptoff, tnu);
            tangentv_h.set(ptoff, tnv);
        }
    }

    if (false_colors) {
        // Pseudo Colors:
        MatrixXd C;
        igl::parula(H,true,C);
        GA_RWHandleV3   color_h(gdp->addFloatTuple(GA_ATTRIB_POINT, "Cd", 3));
        GA_FOR_ALL_PTOFF(gdp, ptoff) { 
            const GA_Index ptidx = gdp->pointIndex(ptoff);  
            UT_ASSERT((uint)ptidx < C.rows());
            const UT_Vector3 cd(C((uint)ptidx, 0), C((uint)ptidx, 1), C((uint)ptidx, 2));
            color_h.set(ptoff, cd);
        }
    }
}

void compute_gradient(GU_Detail *gdp, const GA_ROHandleF &sourceAttrib, MatrixXd &V, MatrixXi &F)
{
    const uint numPoints = gdp->getNumPoints();
    VectorXd U(numPoints);
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        const float val      = sourceAttrib.get(ptoff);
        const GA_Index ptidx = gdp->pointIndex(ptoff);
        U((uint)ptidx) = val;
    }

    // Compute gradient operator: #F*3 by #V
    Eigen::SparseMatrix<double> G;
    igl::grad(V,F,G);

    // Compute gradient of U
    MatrixXd GU = Eigen::Map<const MatrixXd>((G*U).eval().data(),F.rows(),3);
    // Compute gradient magnitude
    const VectorXd GU_mag = GU.rowwise().norm();


    // Copy attributes to Houdini
    { 
        GA_RWHandleV3  gradAttrib_h(gdp->addFloatTuple(GA_ATTRIB_POINT, "gradientAttrib", 3));
        GA_Offset ptoff;
        GA_FOR_ALL_PTOFF(gdp, ptoff) { 
            const GA_Index ptidx = gdp->pointIndex(ptoff);  
            UT_ASSERT((uint)ptidx < GU_mag.rows());
            UT_ASSERT((uint)ptidx < GU.rows());
            UT_Vector3 grad(GU((uint)ptidx, 0),  GU((uint)ptidx, 1), GU((uint)ptidx, 2));
            const float gmag = GU_mag((uint)ptidx, 0);
            grad *= gmag;
            grad.normalize();//?
            gradAttrib_h.set(ptoff, grad);
        }
    }
}

int compute_laplacian(const MatrixXd &V, const MatrixXi &F, 
                      const Eigen::SparseMatrix<double> &L, MatrixXd &U)
{
    Eigen::SparseMatrix<double> M;
    igl::massmatrix(U,F,igl::MASSMATRIX_TYPE_BARYCENTRIC, M);

    // Solve (M-delta*L) U = M*U
    const auto & S = (M - 0.001*L);
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(S);

    if(solver.info() != Eigen::Success)
      return solver.info();
    U = solver.solve(M*U).eval();
    return Eigen::Success;
}

int minimize_area(const MatrixXd &V, const MatrixXi &F,
	const VectorXi& b, MatrixXd &Z)
{
	// L = Laplacian
	// boundaryAttrib = point attribute for fixed points
	// TODO Find and implement minimizing function to perform area minimization (Lagrange Equation)
	// TODO apply boundaries
	// TODO MAYBE to match libigl, convert boundaryAttrib to vector. as in: VectorXi b(n_cnstr_points,1); 
	//from libigl tutorial: https://github.com/libigl/libigl/blob/master/tutorial/304_LinearEqualityConstraints/main.cpp

	// Construct Laplacian and mass matrix
	Eigen::SparseMatrix<double> L, M, Minv, Q;
	VectorXi all_dims;
	igl::colon<int>(0, V.cols() - 1, all_dims);

	igl::cotmatrix(V, F, L);

	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);

	igl::invert_diag(M, Minv);
	// Bi-Laplacian

	////////// from https://github.com/libigl/libigl/blob/master/tutorial/303_LaplaceEquation/main.cpp
	// List of all vertex indices
	VectorXi all, in;
	VectorXi IA; // removed b and IC from original
	igl::colon<int>(0, V.rows() - 1, all);
	// List of interior indices
	igl::setdiff(all, b, in, IA);

	// Construct and slice up Laplacian
	Eigen::SparseMatrix<double> L_in_in, L_in_b;
	//end of 303_LaplaceEquation

	Q = L * (Minv * L);
	// Zero linear term

	MatrixXd B = MatrixXd::Zero(V.rows(), 3);

	MatrixXd bc(b.size(), 3);
	igl::slice(V, b, all_dims, bc);

	// Alternative, short hand
	igl::min_quad_with_fixed_data<double> mqwf;
	// Empty constraints
	//VectorXd Z, Z_const;

	VectorXd Beq;
	Eigen::SparseMatrix<double> Aeq;

	//std::cout << "b: " << b<< "\n";
	std::cout <<"b.size(): "<< b.size() <<"\n";
	std::cout << "bc.size(): " << bc.size() << "\n";
	//std::cout << "bc: " << bc << "\n";

	igl::slice(L, in, in, L_in_in);
	igl::slice(L, in, b, L_in_b);

	// Solve PDE
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > solver(-L_in_in);

	MatrixXd Z_in = solver.solve(L_in_b*bc);
	// slice into solution
	Z = V;
	MatrixXd V_in_b(b.size(), 3);
	/*
	std::cout << "Z_in: " << Z_in<<"\n\n";
	std::cout << "Z: " << Z_in << "\n\n";

	std::cout << "Z.size(): " << Z.size() << "\n";
	std::cout << "in: " << in << "\n";
	std::cout << "slice back solution " << "\n";
	*/
	igl::slice_into(Z_in, in, all_dims, Z);

	igl::slice(V, in, all_dims, V_in_b);

	//alternative 
	/*
	igl::min_quad_with_fixed_precompute(Q, b, Aeq, true, mqwf);
	std::cout << i << "\n";
	i++;
	igl::min_quad_with_fixed_solve(mqwf, B, bc, Beq, Z);
	std::cout << i << "\n";
	i++;
	
	*/
	return Eigen::Success;
}

} // end of SOP_IGL namespce

void copy_back_to_houdini(GU_Detail *gdp, MatrixXd &V)
{
	// Copy back to Houdini:
	GA_Offset ptoff;
	GA_FOR_ALL_PTOFF(gdp, ptoff)
	{
		const GA_Index ptidx = gdp->pointIndex(ptoff);
		if ((uint)ptidx < V.rows())
		{
			const UT_Vector3 pos(V((uint)ptidx, 0),
				V((uint)ptidx, 1),
				V((uint)ptidx, 2));
			gdp->setPos3(ptoff, pos);
		}
	}
}
void constrain_matrix_by_attrib(GU_Detail *gdp, const GA_ROHandleF& sourceAttrib, Eigen::SparseMatrix<double> &mat)
{
	std::vector<Eigen::Triplet<double>> triplets;
	GA_Offset ptoffs;
	int i = 0;
	int n_rows = mat.rows();
	GA_FOR_ALL_PTOFF(gdp, ptoffs) {
		if (sourceAttrib.get(ptoffs) == 1.0) {
			for (int j = 0; j<n_rows; ++j) {
				triplets.push_back({ i,j,0.0 });
				triplets.push_back({ j,i,0.0 });
			}
			triplets.push_back({ i,i,1.0 });
		}
		i++;
	}
	//if(i==gdp.numPoints()) TODO Print warning!
	mat.setFromTriplets(triplets.begin(), triplets.end());
}

OP_Node *
SOP_IGLDiscreteGeometry::myConstructor(OP_Network *net, const char *name, OP_Operator *op)
{
    return new SOP_IGLDiscreteGeometry(net, name, op);
}

SOP_IGLDiscreteGeometry::SOP_IGLDiscreteGeometry(OP_Network *net, const char *name, OP_Operator *op)
    : SOP_Node(net, name, op), myGroup(NULL)
{
   
    mySopFlags.setManagesDataIDs(true);
}

SOP_IGLDiscreteGeometry::~SOP_IGLDiscreteGeometry() {}

OP_ERROR
SOP_IGLDiscreteGeometry::cookInputGroups(OP_Context &context, int alone)
{
    
    return cookInputPointGroups(
        context, // This is needed for cooking the group parameter, and cooking the input if alone.
        myGroup, // The group (or NULL) is written to myGroup if not alone.
        alone,   // This is true iff called outside of cookMySop to update handles.
                 // true means the group will be for the input geometry.
                 // false means the group will be for gdp (the working/output geometry).
        true,    // (default) true means to set the selection to the group if not alone and the highlight flag is on.
        0,       // (default) Parameter index of the group field
        -1,      // (default) Parameter index of the group type field (-1 since there isn't one)
        true,    // (default) true means that a pointer to an existing group is okay; false means group is always new.
        false,   // (default) false means new groups should be unordered; true means new groups should be ordered.
        true,    // (default) true means that all new groups should be detached, so not owned by the detail;
                 //           false means that new point and primitive groups on gdp will be owned by gdp.
        0        // (default) Index of the input whose geometry the group will be made for if alone.
    );
}

OP_ERROR
SOP_IGLDiscreteGeometry::cookMySop(OP_Context &context)
{
	std::cout << "cook\n";
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();

    fpreal t = context.getTime();
    duplicateSource(0, context);

    // Copy to eigen.
    gdp->convex(); // only triangles for now.
    uint numPoints = gdp->getNumPoints();
    uint numPrims  = gdp->getNumPrimitives();
    MatrixXd V(numPoints, 3); // points
    MatrixXi F(numPrims, 3); // faces
	MatrixXd Z;
    SOP_IGL::detail_to_eigen(*gdp, V, F);

    /*    Laplacian smoothing   */
    const float laplacian = LAPLACIAN(t);

    if (laplacian != 0)
    {
        int laplacian_iterations = (int)ceil(laplacian);
        float laplacian_ratio    = laplacian - floorf(laplacian);
        laplacian_ratio = laplacian_ratio != 0.f ? laplacian_ratio : 1.f;

        // Start the interrupt server
        UT_AutoInterrupt boss("Laplacian smoothing...");
        Eigen::SparseMatrix<double> L;
        // Compute Laplace-Beltrami operator: #V by #V
        igl::cotmatrix(V,F,L);
        // Smoothing:
        MatrixXd U; U = V;
        MatrixXd T;
        T = MatrixXd::Zero(V.rows(), V.cols());

        while(laplacian_iterations) 
        {
            // User interaption/
            if (boss.wasInterrupted())
                return error();

            if (SOP_IGL::compute_laplacian(V, F, L, U) != Eigen::Success) {
                addWarning(SOP_MESSAGE, "Can't compute laplacian with current geometry.");
                return error();
            }
            laplacian_iterations--;

            // if (laplacian_iterations > 0)
            //     T += (U - T);
            // else
            //     T += (U - T) * laplacian_ratio;
            T = U;
        }


    }
	
	else if (AREA_MIN(t) != 0)
	{
		//std::cout << "Before area minimizing\n";
		//get fixed points
		GA_ROHandleF borderAttrib(gdp, GA_ATTRIB_POINT, "border");
		if(borderAttrib.isValid())
		{
			std::cout << "got attrib, processing\n";
			//b=boundary pts
			VectorXi b;
			VectorXd bc;
			std::cout << "boundary attrib\n";
			SOP_IGL::boundaryAttrib_to_eigen(*gdp, b, borderAttrib);
			std::cout << "minimizing area\n";
			SOP_IGL::minimize_area(V, F, b, Z);

			//std::cout << "completed func\n";
			//std::cout << "b.size():\n" << b.size();
			//std::cout << "Z.size():\n" << Z.size();
			//std::cout << "V.size():\n" << V.size() << "\n";

			// Copy back to Houdini:
			copy_back_to_houdini(gdp, Z);
		}
	}

	if (FAIRING(t) != 0)
	{

		//std::cout << "Before area minimizing\n";
		//get fixed points
		//TODO convert process to use prim group
		GA_ROHandleF borderAttrib(gdp, GA_ATTRIB_POINT, "boundary");
		if (borderAttrib.isValid())
		{
			std::cout << "got attrib, processing\n";
			// b=boundary pts, bc= boundary constraints (b positions)
			VectorXi b;
			std::cout << "boundary attrib\n";
			SOP_IGL::boundaryAttrib_to_eigen(*gdp, b, borderAttrib);
			//get boundary positions into bc
			//TODO separate bc creation to func
			MatrixXd bc(b.size(), 3);
			VectorXi all_dims;
			igl::colon<int>(0, V.cols() - 1, all_dims);
			igl::slice(V, b, all_dims, bc);
			//solve
			int k = 2;
			if (igl::harmonic(V, F, b, bc, k, Z)) {
				// Copy back to Houdini:
				copy_back_to_houdini(gdp, Z);
			}
			else{
				std::cout << "failed to solve mesh fairing\n";
			}

		}
	}

    gdp->getP()->bumpDataId();
    return error();
}
