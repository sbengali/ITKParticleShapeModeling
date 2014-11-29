/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkPSMParticleEntropyFunction_hxx
#define __itkPSMParticleEntropyFunction_hxx
#include "itkPSMParticleEntropyFunction.h"

namespace itk {

template <class TGradientNumericType, unsigned int VDimension>
TGradientNumericType
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::AngleCoefficient( const GradientVectorType& p_i_normal,  const GradientVectorType& p_j_normal) const
{
    // Get the cosine of the angle between the two particles' normals
    TGradientNumericType cosine = dot_product(p_i_normal,p_j_normal) /
            (p_i_normal.magnitude()*p_j_normal.magnitude() + 1.0e-6);

    // the flat region
    if ( cosine >= m_FlatCutoff ) return 1.0;

    // the feathered region
    return ( cos( (m_FlatCutoff - cosine) / (1.0+m_FlatCutoff) * (3.14159265358979/2.0) )) ;
} 

template <class TGradientNumericType, unsigned int VDimension>
void
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::ComputeAngularWeights(const PointType &pos,
                        const typename ParticleSystemType::PointVectorType &neighborhood,
                        const PSMImageDomainWithGradients<TGradientNumericType, VDimension> *domain,
                        std::vector<double> &weights) const
{
    GradientVectorType posnormal = domain->SampleNormalVnl(pos, 1.0e-10);
    weights.resize(neighborhood.size());

    for (unsigned int i = 0; i < neighborhood.size(); i++)
    {
        weights[i] = this->AngleCoefficient(posnormal,
                                            domain->SampleNormalVnl(neighborhood[i].Point, 1.0e-10));
        if (weights[i] < 1.0e-5) weights[i] = 0.0;
    }
}

template <class TGradientNumericType, unsigned int VDimension>
double
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::EstimateSigma(unsigned int, const typename ParticleSystemType::PointVectorType &neighborhood,
                const std::vector<double> &weights,
                const PointType &pos,  double initial_sigma,  double precision,
                int &err) const
{
    const double epsilon = 1.0e-5;
    const double min_sigma = 1.0e-4;

    const double M = static_cast<double>(VDimension);
    const double MM = M * M * 2.0 + M;

    double error = 1.0e6;
    double sigma, prev_sigma;
    sigma = initial_sigma;

    while (error > precision)
    {
        VectorType r_vec;
        double A = 0.0;
        double B = 0.0;
        double C = 0.0;
        double sigma2 = sigma * sigma;
        double sigma22 = sigma2 * 2.0;

        for (unsigned int i = 0; i < neighborhood.size(); i++)
        {
            if (weights[i] < epsilon) continue;

            //    if ( neighborhood[i].Index == idx) continue;
            for (unsigned int n = 0; n < VDimension; n++)
            {
                r_vec[n] = pos[n] - neighborhood[i].Point[n];
            }

            double r = r_vec.magnitude();
            double r2 = r*r;
            double alpha = exp(-r2 / sigma22) * weights[i];
            A += alpha;
            B += r2 * alpha;
            C += r2 * r2 * alpha;
        } // end for i

        prev_sigma = sigma;

        if (A < epsilon)
        {
            err = 1;
            return sigma;
        }; // results are not meaningful

        // First order convergence update.  This is a fixed point iteration.
        //sigma = sqrt(( 1.0 / DIM ) * ( B / A));

        // Second order convergence update (Newton-Raphson).  This is the first
        // derivative of the negative of the probability density estimation function squared over the
        // second derivative.

        // old math
        //    sigma -= (sigma2 * VDimension * A * A - A  * B) / (((2.0 * sigma * VDimension) * A * A -
        //                                          (1.0/(sigma2*sigma))*(A*C-B*B)) + epsilon);

        // New math -- results are not obviously different?
        sigma -= (A * (B - A * sigma2 * M)) /
                ( (-MM * A *A * sigma) - 3.0 * A * B * (1.0 / (sigma + epsilon))
                  - (A*C + B*B) * (1.0 / (sigma2 * sigma + epsilon)) + epsilon);

        error = 1.0 - fabs((sigma/prev_sigma));

        // Constrain sigma.
        if (sigma < min_sigma)
        {
            sigma = min_sigma;
            error = precision; // we are done if sigma has vanished
        }
        else
        {
            if (sigma < 0.0) sigma = min_sigma;
        }

    } // end while (error > precision)

    err = 0;
    return sigma;

} // end estimate sigma


template <class TGradientNumericType, unsigned int VDimension>
typename PSMParticleEntropyFunction<TGradientNumericType, VDimension>::VectorType
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::Evaluate(unsigned int idx,unsigned int d, const ParticleSystemType * system,
           double &maxdt, double &energy) const
{
    if(m_PairwisePotentialType == "gaussian")
        return this->EvaluateGaussian(idx, d, system, maxdt, energy);
    else if(m_PairwisePotentialType == "cotan")
        return this->EvaluateCotan(idx, d, system, maxdt, energy);

}

template <class TGradientNumericType, unsigned int VDimension>
typename PSMParticleEntropyFunction<TGradientNumericType, VDimension>::VectorType
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::EvaluateGaussian(unsigned int idx,unsigned int d, const ParticleSystemType * system,
                   double &maxdt, double &energy) const
{
    // Grab a pointer to the domain.  We need a Domain that has surface normal information.
    const PSMImageDomainWithGradients<TGradientNumericType, VDimension> *
            domain = static_cast<const PSMImageDomainWithGradients<
            TGradientNumericType, VDimension> *>(system->GetDomain(d));
    const double epsilon = 1.0e-6;

    // Retrieve the previous optimal sigma value for this point.  If the value is
    // tiny (i.e. unitialized) then use a fraction of the maximum allowed
    // neighborhood radius.
    double sigma = m_SpatialSigmaCache->operator[](d)->operator[](idx);
    if (sigma < epsilon)
    { sigma = m_MinimumNeighborhoodRadius / m_NeighborhoodToSigmaRatio;}

    // Determine the extent of the neighborhood that will be used in the Parzen
    // windowing estimation.  The neighborhood extent is based on the optimal
    // sigma calculation and limited to a user supplied maximum radius (probably
    // the size of the domain).
    double neighborhood_radius = sigma * m_NeighborhoodToSigmaRatio;
    if (neighborhood_radius > m_MaximumNeighborhoodRadius)
    { neighborhood_radius = m_MaximumNeighborhoodRadius; }

    // Get the position for which we are computing the gradient.
    PointType pos = system->GetPosition(idx, d);

    // Get the neighborhood surrounding the point "pos".
    typename ParticleSystemType::PointVectorType neighborhood
            = system->FindNeighborhoodPoints(pos, neighborhood_radius, d);

    // Compute the weights based on angle between the neighbors and the center.
    std::vector<double> weights;
    this->ComputeAngularWeights(pos,neighborhood,domain,weights);

    // Estimate the best sigma for Parzen windowing.  In some cases, such as when
    // the neighborhood does not include enough points, the value will be bogus.
    // In these cases, an error != 0 is returned, and we try the estimation again
    // with an increased neighborhood radius.
    int err;
    sigma =  this->EstimateSigma(idx, neighborhood,weights,pos, sigma, epsilon, err);

    while (err != 0)
    {
        neighborhood_radius *= 2.0;
        // Constrain the neighborhood size.  If we have reached a maximum
        // possible neighborhood size, we'll just go with that.
        if ( neighborhood_radius > this->GetMaximumNeighborhoodRadius())
        {
            sigma = this->GetMaximumNeighborhoodRadius() / this->GetNeighborhoodToSigmaRatio();
            neighborhood_radius =  this->GetMaximumNeighborhoodRadius();
            break;
        }
        else
        {
            sigma = neighborhood_radius / this->GetNeighborhoodToSigmaRatio();
        }

        neighborhood = system->FindNeighborhoodPoints(pos, neighborhood_radius, d);
        this->ComputeAngularWeights(pos,neighborhood,domain,weights);
        sigma = this->EstimateSigma(idx, neighborhood, weights, pos, sigma, epsilon, err);
    } // done while err

    // Constrain sigma to a maximum reasonable size based on the user-supplied
    // limit to neighborhood size.
    if (sigma > this->GetMaximumNeighborhoodRadius())
    {
        sigma = this->GetMaximumNeighborhoodRadius() / this->GetNeighborhoodToSigmaRatio();
        neighborhood_radius = this->GetMaximumNeighborhoodRadius();
        neighborhood = system->FindNeighborhoodPoints(pos, neighborhood_radius, d);
        this->ComputeAngularWeights(pos,neighborhood,domain,weights);
    }

    //  std::cout << idx <<  "\t SIGMA = " << sigma << "\t NEIGHBORHOOD SIZE = " << neighborhood.size()
    //            << "\t NEIGHBORHOOD RADIUS= " << neighborhood_radius << std::endl;

    // We are done with the sigma estimation step.  Cache the sigma value for
    // next time.
    m_SpatialSigmaCache->operator[](d)->operator[](idx) = sigma;

    //----------------------------------------------

    // Compute the gradients.
    double sigma2inv = 1.0 / (2.0* sigma * sigma + epsilon);

    VectorType r;
    VectorType gradE;

    for (unsigned int n = 0; n < VDimension; n++)
    {
        gradE[n] = 0.0;
    }

    double A = 0.0;
    for (unsigned int i = 0; i < neighborhood.size(); i++)
    {
        //    if ( neighborhood[i].Index == idx) continue;
        if (weights[i] < epsilon) continue;

        for (unsigned int n = 0; n < VDimension; n++)
        {
            // Note that the Neighborhood object has already filtered the
            // neighborhood for points whose normals differ by > 90 degrees.
            r[n] = pos[n] - neighborhood[i].Point[n];
        }

        double q = exp( -dot_product(r, r) * sigma2inv);
        A += q;

        for (unsigned int n = 0; n < VDimension; n++)
        {
            gradE[n] += weights[i] * r[n] * q;
        }
    }

    double p = 0.0;
    if (A > epsilon)
    {
        p = -1.0 / (A * sigma * sigma);
    }

    // TEST
    //   vnl_vector_fixed<float, VDimension> tosurf = domain->SampleGradientVnl(pos);
    //   float tosurfmag = tosurf.magnitude() + 1.0e-5;

    // end test
    //   float f = domain->Sample(pos);

    for (unsigned int n = 0; n <VDimension; n++)
    {
        gradE[n] *= p;
        // TEST
        //     gradE[n] += f * (tosurf[n] / tosurfmag);
        // end test
    }
    //   maxdt = sigma * sigma;
    maxdt = 0.5;

    energy = (A * sigma2inv );

    return gradE;
}

template <class TGradientNumericType, unsigned int VDimension>
typename PSMParticleEntropyFunction<TGradientNumericType, VDimension>::VectorType
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::EvaluateCotan(unsigned int idx,unsigned int d, const ParticleSystemType * system,
                double &maxdt, double &energy) const
{
    // Grab a pointer to the domain.  We need a Domain that has surface normal information.
    const PSMImageDomainWithGradients<TGradientNumericType, VDimension> *
            domain = static_cast<const PSMImageDomainWithGradients<
            TGradientNumericType, VDimension> *>(system->GetDomain(d));

    const double epsilon           = 1.0e-6;
    const double NBHD_SIGMA_FACTOR = 1.5f;

    // Determine the extent of the neighborhood that will be used in the Parzen
    // windowing estimation.
    double neighborhood_radius = m_GlobalSigma * NBHD_SIGMA_FACTOR * this->GetNeighborhoodToSigmaRatio();

    if (neighborhood_radius > this->GetMaximumNeighborhoodRadius())
        neighborhood_radius = this->GetMaximumNeighborhoodRadius();


    // Get the position for which we are computing the gradient.
    PointType pos = system->GetPosition(idx, d);

    // Get the neighborhood surrounding the point "pos".
    typename ParticleSystemType::PointVectorType neighborhood
            = system->FindNeighborhoodPoints(pos, neighborhood_radius, d);

    // Compute the weights based on angle between the neighbors and the center.
    std::vector<double> weights;
    this->ComputeAngularWeights(pos,neighborhood,domain,weights);

    // Compute the gradients.
    VectorType r;
    VectorType gradE;
    double     rmag;

    for (unsigned int n = 0; n < VDimension; n++)
    {
        gradE[n] = 0.0;
    }

    double prob_xi = epsilon;
    double M       = epsilon;
    for (unsigned int k = 0; k < neighborhood.size(); k++)
    {
        for (unsigned int n = 0; n < VDimension; n++)
        {
            // Note that the Neighborhood object has already filtered the
            // neighborhood for points whose normals differ by > 90 degrees.
            r[n] = (pos[n] - neighborhood[k].Point[n]) ;
        }
        rmag = r.magnitude();

        double dPhi = this->ComputeModifiedCotangentDerivative(rmag);
        for (unsigned int n = 0; n < VDimension; n++)
        {
            gradE[n] += ( ( weights[k] * dPhi * r[n] )/(rmag + epsilon) );
        }

        prob_xi += weights[k] * this->ComputeModifiedCotangent(rmag);
        M       += weights[k];
    }

    prob_xi /= M;
    for (unsigned int n = 0; n < VDimension; n++)
    {
        gradE[n] *= ( (-1.0/ (M * prob_xi ) ) );
    }


    maxdt   = m_GlobalSigma * 0.1;
    energy  = prob_xi ;

    return gradE ;
}

template <class TGradientNumericType, unsigned int VDimension>
void
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::EstimateGlobalSigma(const ParticleSystemType * system)
{
    const double epsilon = 1.0e-6;
    const int K          = 6; // hexagonal ring - one jump

    const unsigned int num_samples   = system->GetNumberOfDomains(); // num_shapes * domains_per_shape
    const unsigned int num_particles = system->GetNumberOfParticles();
    const unsigned int num_shapes    = num_samples / m_DomainsPerShape;

    if (num_particles < 7)
    {
        m_GlobalSigma = this->GetMaximumNeighborhoodRadius() / this->GetNeighborhoodToSigmaRatio();
        return;
    }

    double avgDist = 0.0;
    for (unsigned int domainInShape = 0; domainInShape < m_DomainsPerShape; domainInShape++)
    {
        for (unsigned int shapeNo = 0; shapeNo < num_shapes; shapeNo++)
        {
            // linear index of the current domain in the particle system
            int dom = shapeNo * m_DomainsPerShape + domainInShape;

            // get the particles of the current domain in shape sample to construct a sample for kdtree
            typename SampleType::Pointer sample = SampleType::New();
            sample->SetMeasurementVectorSize( VDimension );
            for (unsigned int idx = 0; idx < num_particles ; idx++)
            {
                // get the current particle
                PointType            pos  = system->GetPosition(idx, dom);

                // add it to the sample
                MeasurementVectorType mv;
                for(unsigned int n = 0 ; n < VDimension; n++)
                    mv[n] = pos[n];

                sample->PushBack( mv );
            }

            typename TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
            treeGenerator->SetSample( sample );
            treeGenerator->SetBucketSize( 16 );
            treeGenerator->Update();

            typename TreeType::Pointer tree = treeGenerator->GetOutput();

            // Find the closest points to each particle
            for (unsigned int idx = 0; idx < num_particles ; idx++)
            {
                // get the current particle
                PointType  pos  = system->GetPosition(idx, dom);

                // construct a query point
                MeasurementVectorType queryPoint;
                for(unsigned int n = 0 ; n < VDimension; n++)
                    queryPoint[n] = pos[n];

                // K-Neighbor search
                unsigned int numberOfNeighbors = K+1; // +1 to exclude myself
                typename TreeType::InstanceIdentifierVectorType neighbors;
                tree->Search( queryPoint, numberOfNeighbors, neighbors ) ;

                double meanDist = 0;
                for ( unsigned int k = 0 ; k < neighbors.size() ; k++ )
                {
                    // the distance to myself will be zero anyway, no need to explicitly check for this
                    // only ignore it when computing the average

                    MeasurementVectorType neighPoint = tree->GetMeasurementVector( neighbors[k] );
                    double curDist = 0.0;
                    for(unsigned int n = 0 ; n < VDimension; n++)
                        curDist += pow(queryPoint[n]-neighPoint[n],2.0);
                    curDist = sqrt(curDist);

                    meanDist += curDist;
                }
                meanDist /= K; // excluding myself from the mean computation
                avgDist += meanDist;
            }
        }
    }

    avgDist /= (double)(num_particles * num_shapes * m_DomainsPerShape);

    m_GlobalSigma = avgDist / 0.57f; // based on hexagonal packing, sigma is the distance to the second ring

    if (m_GlobalSigma < epsilon)
    {
        m_GlobalSigma = this->GetMinimumNeighborhoodRadius() / this->GetNeighborhoodToSigmaRatio();
    }
}

template <class TGradientNumericType, unsigned int VDimension>
void
PSMParticleEntropyFunction<TGradientNumericType, VDimension>
::BeforeIteration()
{
    if(m_PairwisePotentialType == "cotan")
    {
        if(m_GlobalSigma < 0.0)
        {
            // only compute sigma once during the optimization of a specific scale
            this->EstimateGlobalSigma(this->GetParticleSystem());
            std::cout << "GlobalSigma: " << m_GlobalSigma << std::endl;
        }
        else
        {
            // compute the global sigma for the whole particle system using its current status (particles position)
            double oldSigma = m_GlobalSigma;
            this->EstimateGlobalSigma(this->GetParticleSystem());

            // make sure that we update the global sigma at the beginning (in the constructor, it is -1)
            if ( (abs(oldSigma - m_GlobalSigma)/m_GlobalSigma) < 0.1)
            {
                // not that much change, probably same number of particles, don't change the global sigma
                m_GlobalSigma = oldSigma;
            }
        }

    }

}

}// end namespace
#endif
