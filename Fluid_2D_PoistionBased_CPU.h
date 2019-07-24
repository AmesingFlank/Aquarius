//
// Created by AmesingFlank on 2019-07-21.
//

#ifndef AQUARIUS_FLUID_2D_POISTIONBASED_CPU_H
#define AQUARIUS_FLUID_2D_POISTIONBASED_CPU_H


#include "Fluid_2D_PCISPH.cuh"

#undef PRINT_INDEX

#define PRINT_INDEX 980

namespace Fluid_2D_PositionBased_CPU {

    using Particle = Fluid_2D_PCISPH::Particle;

    class Fluid : public Fluid_2D {
    public:


        const float gridBoundaryX = 10.f;
        const float gridBoundaryY = 5.f;


        const float restDensity = 100;
        const float particleRadius = sqrt(1 / restDensity / M_PI);
        const float SCP = 0.7;
        const float kernelRadius = sqrt(4.0 / (M_PI * restDensity * SCP));


        const float cellPhysicalSize = kernelRadius;

        const int gridSizeX = ceil(gridBoundaryX / cellPhysicalSize);
        const int gridSizeY = ceil(gridBoundaryY / cellPhysicalSize);
        const int cellCount = gridSizeX * gridSizeY;


        int *cellStart = new int[cellCount];
        int *cellEnd = new int[cellCount];

        std::vector <Particle> particles;
        int particleCount = 0;


        float precomputedSpacing;

        const float timeStep = 0.001;


        void computeSpacing() {
            float l = 0;
            float r = kernelRadius;
            while (r - l > 0) {
                float m = (l + r) / 2;
                float contribPerParticle = poly6(make_float2(m, 0), kernelRadius);
                float totalDensity = 4 * contribPerParticle + poly6(make_float2(0, 0), kernelRadius);
                if (totalDensity == restDensity) break;
                if (totalDensity > restDensity) {
                    l = m;
                } else {
                    r = m;
                }
            }
            precomputedSpacing = (l + r) / 2;
            std::cout << "precomputed spacing: " << precomputedSpacing;
            std::cout << "  which is  " << precomputedSpacing / particleRadius << "  times particle radius"
                      << std::endl;
        }

        Fluid () {
            std::cout << "kernal radius " << kernelRadius << std::endl;
            std::cout << "particle radius " << particleRadius << std::endl;

            std::cout << "gridSizeX: " << gridSizeX << "     gridSizeY:" << gridSizeY << std::endl;
            std::cout << "self contributed density: " << poly6(make_float2(0, 0), kernelRadius) << std::endl;

            computeSpacing();
            initFluid();
        }

        void performSpatialHashing(int version = 0) {


            calcHashImpl();

            auto cmp = [&](Particle &a, Particle &b) -> bool {
                return a.hash < b.hash;
            };

            std::sort(particles.begin(), particles.end(), cmp);


            memset(cellStart, -1, cellCount * sizeof(*cellStart));
            memset(cellEnd, -1, cellCount * sizeof(*cellEnd));

            findCellStartEndImpl();


        }


        void initFluid() {


            for (float x = 0; x < gridBoundaryX; x += precomputedSpacing) {
                for (float y = 0; y < gridBoundaryY; y += precomputedSpacing) {

                    float2 pos = make_float2(x, y);
                    if (pos.y < gridBoundaryY * 0.53 && pos.x < gridBoundaryX * 1.5) {
                        particles.emplace_back(pos);
                    } else if (pow(pos.x - 0.5 * gridBoundaryX, 2) + pow(pos.y - 0.7 * gridBoundaryY, 2) <=
                               pow(0.2 * gridBoundaryY, 2)) {
                        //particles.emplace_back(pos);
                    }
                }
            }
            particleCount = particles.size();
            std::cout << "particles:" << particleCount << std::endl;

            for (int i = 0; i < cellCount; ++i) {
                cellStart[i] = -1;
                cellEnd[i] = -1;
            }


        }

        void simulationStep(float totalTime) {

            performSpatialHashing();
            calcOtherForces();
            calcPositionVelocity();
            for (int i = 0; i < 10; ++i) {


                calcDensity();

                calcLambda();

                updatePosition();

            }

            commitPositionVelocity();

            updateTexture();
            std::cout << "\n --------------------finished one step---------------- \n" << std::endl;
        }


        virtual void updateTexture() override {
            printGLError();
            glBindTexture(GL_TEXTURE_2D, texture);
            int texSizeX = 256;
            int texSizeY = 126;
            float texCellPhysicalSize = cellPhysicalSize * gridSizeX / texSizeX;
            size_t imageSize = texSizeX * texSizeY * 4 * sizeof(unsigned char);
            unsigned char *image = (unsigned char *) malloc(imageSize);
            memset(image, 255, imageSize);


            for (int index = 0; index < particleCount; ++index) {
                Particle &particle = particles[index];
                int cellX = particle.position.x / texCellPhysicalSize;
                int cellY = particle.position.y / texCellPhysicalSize;
                int cellID = cellY * texSizeX + cellX;

                cellID = max(0, min(cellID, texSizeX * texSizeY - 1));

                unsigned char *base = image + cellID * 4;


                //std::cout<<index<<std::endl;
                if (index == PRINT_INDEX || base[0] == 254) {
                    base[0] = 254;
                    base[1] = 0;
                    base[2] = 0;
                    base[3] = 255;
                    continue;
                }

                base[0] = 0;
                base[1] = 0;
                base[2] = 255;
                base[3] = 255;
            }


            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texSizeX, texSizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
            glGenerateMipmap(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);
            free(image);
            printGLError();

        }


        void calcHashImpl() {
            for (int index = 0; index < particleCount; ++index) {
                Particle &particle = particles[index];

                float2 pos = particle.position;
                int x = pos.x / cellPhysicalSize;
                int y = pos.y / cellPhysicalSize;

                int hash = x * gridSizeY + y;

                particle.hash = hash;

            }
        }

        void findCellStartEndImpl() {
            for (int index = 0; index < particleCount; ++index) {
                Particle &particle = particles[index];

                int cellID = particle.hash;

                if (index == 0 || particles[index - 1].hash < cellID) {
                    cellStart[cellID] = index;
                }

                if (index == particleCount - 1 || particles[index + 1].hash > cellID) {
                    cellEnd[cellID] = index;
                }

            }
        }

        void calcOtherForces() {
            for (Particle &particle:particles) {
                particle.otherForces = make_float2(0, -9.8);
            }
        }

        void calcPositionVelocity() {
            for (Particle &particle:particles) {

                float2 acc = particle.otherForces + particle.pressureForce;

                particle.newVelocity = particle.velocity + timeStep * acc;

                float2 meanVelocity = (particle.newVelocity + particle.velocity) / 2;

                float2 newPosition = particle.position + meanVelocity * timeStep;

                if (newPosition.x < 0 || newPosition.x > gridBoundaryX) {
                    particle.newVelocity.x = 0;
                }

                if (newPosition.y < 0 || newPosition.y > gridBoundaryY) {
                    particle.newVelocity.y = 0;
                }

                newPosition.x = min(gridBoundaryX - 1e-6, max(0.0, newPosition.x));
                newPosition.y = min(gridBoundaryY - 1e-6, max(0.0, newPosition.y));


                particle.newPosition = newPosition;
            }
        }

        void commitPositionVelocity() {
            for (Particle &particle:particles) {
                particle.velocity = (particle.newPosition - particle.position) / timeStep;
                particle.position = particle.newPosition;
            }
        }

        void calcDensity() {
            for (Particle &particle:particles) {
                int centerCell = particle.hash;
                int cellsToCheck[9];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        cellsToCheck[c * 3 + r] = centerCell + (c - 1) * gridSizeY + (r - 1);
                    }
                }
                float density = 0;
                for (int cellID :cellsToCheck) {

                    if (cellID < 0 || cellID >= cellCount) continue;

                    for (int j = cellStart[cellID]; j <= cellEnd[cellID]; ++j) {
                        if (j == -1) continue;
                        Particle &that = particles[j];
                        density += poly6(particle.newPosition - that.newPosition, kernelRadius);
                    }
                }

                if (particle.newPosition.x <= 0 || particle.newPosition.x >= gridBoundaryX - 1e-6) {
                    particle.density *= 2;
                }

                if (particle.newPosition.y <= 0 || particle.newPosition.y >= gridBoundaryY - 1e-6) {
                    particle.density *= 2;
                }

                particle.density = density;

            }
        }

        void calcLambda() {
            for (int index = 0; index < particleCount; ++index) {
                Particle &particle = particles[index];

                float2 grad_pi_Ci = make_float2(0, 0);
                float sum_dot_grad_pj_Ci = 0;


                int centerCell = particle.hash;
                int cellsToCheck[9];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        cellsToCheck[c * 3 + r] = centerCell + (c - 1) * gridSizeY + (r - 1);
                    }
                }

                for (int cellID :cellsToCheck) {

                    if (cellID < 0 || cellID >= cellCount) continue;

                    for (int j = cellStart[cellID]; j <= cellEnd[cellID]; ++j) {
                        if (j == -1) continue;
                        Particle &pj = particles[j];
                        float2 weight = spikey_grad(particle.newPosition - pj.newPosition, kernelRadius);
                        grad_pi_Ci += weight;
                        sum_dot_grad_pj_Ci += dot(weight, weight);
                    }
                }

                float Ci = particle.density / restDensity - 1;

                if (index == PRINT_INDEX) {
                    std::cout << "Ci: " << Ci << std::endl;
                }

                float denominator = (sum_dot_grad_pj_Ci + dot(grad_pi_Ci, grad_pi_Ci) + 0.00001) / restDensity;

                float lambda = -Ci / denominator;

                particle.lambda = lambda;

            }
        }


        void updatePosition() {
            for (int index = 0; index < particleCount; ++index) {
                Particle &particle = particles[index];

                float2 deltaPosition = make_float2(0, 0);

                int centerCell = particle.hash;
                int cellsToCheck[9];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        cellsToCheck[c * 3 + r] = centerCell + (c - 1) * gridSizeY + (r - 1);
                    }
                }

                for (int cellID :cellsToCheck) {

                    if (cellID < 0 || cellID >= cellCount) continue;

                    for (int j = cellStart[cellID]; j <= cellEnd[cellID]; ++j) {
                        if (j == -1) continue;
                        Particle &pj = particles[j];
                        float2 weight = spikey_grad(particle.newPosition - pj.newPosition, kernelRadius);
                        deltaPosition += weight * (particle.lambda + pj.lambda) / restDensity;
                    }
                }

                particle.newPosition += deltaPosition;

                particle.newPosition.x = min(gridBoundaryX - 1e-6, max(0.0, particle.newPosition.x));
                particle.newPosition.y = min(gridBoundaryY - 1e-6, max(0.0, particle.newPosition.y));

            }
        }


    };
}


#endif //AQUARIUS_FLUID_2D_POISTIONBASED_CPU_H
