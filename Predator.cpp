// Eden Forbes
// MinCogEco Predator

#include "Predator.h"
#include "Prey.h"
#include "random.h"
#include "CTRNN.h"

// Constants
const double SpaceSize = 5000;
const double HalfSpace = SpaceSize/2;

// *******
// Control
// *******

// Init the agent
void Predator::Set(int networksize, double pred_gain, double pred_s_width, double pred_frate, double pred_feff, double pred_metaloss, double pred_b_thresh, double pred_handling_time, double pred_PCOT_scalar, double pred_NCOT_scalar, double pred_NCOT_thresh)
{
    size = networksize;
	gain = pred_gain; 
    sensorweights.SetBounds(1, 4*size);
	sensorweights.FillContents(0.0);
	pos = 0.0;
	pastpos = 0.0;
	f_sensor = 0.0;
    p_sensor = 0.0;
    h_sensor = 0.0;
    s_width = pred_s_width;
    state = 1.0;
    frate = pred_frate;
    feff = pred_feff;
    metaloss = pred_metaloss;
    death = false;
    birth = false;
    birth_thresh = pred_b_thresh;
    handling_time = pred_handling_time;
    handling_counter = 0.0;
    handling = false;
    // interaction rates
    munchrate = 0.0;
    birthrate = 0.0;
    snackflag = 0.0;
    // Metabolism parameters
    mov_dist = 0.0;
    PCOT = pred_PCOT_scalar;
    NCOT = pred_NCOT_scalar;
    NCOT_thresh = pred_NCOT_thresh;
    // FITNESS TRACKER
    fitness = 0.0;
}

// Reset the state of the agent
void Predator::Reset(double initpos, double initstate)
{
	pos = initpos;
	pastpos = initpos;
    f_sensor = 0.0;
	p_sensor = 0.0;
    h_sensor = 0.0;
    state = initstate;
    death = false;
    birth = false;
	NervousSystem.RandomizeCircuitState(0.0,0.0);
    handling = false;
    handling_counter = 0.0;
}

// Sense 
void Predator::Sense(TVector<double> &prey_loc, TVector<double> &food_pos, bool hutchflag, int hutchL, int hutchR)
{
    // Sense
	double mindistL = 99999;
    double mindistR = 99999;

    for (int i = 0; i < food_pos.Size(); i++){
        double d = food_pos[i] - pos;
        if (d < 0 && d >= -HalfSpace){
            // Closest to the left side, distance is as calculated
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d < 0 && d < -HalfSpace){
            // Closest to the right side, distance is total area + - left side distance 
            d = SpaceSize + d;
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d > 0 && d > HalfSpace){
            // Closest to the left side, distance is -total area + right side distance
            d = -SpaceSize + d;
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d > 0 && d <= HalfSpace){
            // Closest to the right side, distance is as calculated
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d == 0){
            d = 0;
        }
        else{
            printf("Food pos size = %d\n", food_pos.Size());
            printf("Error in predator food sensing\n");
            printf("d = %f\n", d);
        }
    }
    // Cumulate, distance fits in the Gaussian as the difference of the mean (position) and the state (food position)
    // Negate left so negative sensor reading says food left, positive says food right, zero says no food or food on both sides.
	f_sensor = -2*exp(-(mindistL) * (mindistL) / (2 * s_width * s_width)) + 2*exp(-(mindistR) * (mindistR) / (2 * s_width * s_width));
    // printf("Prey food sensor = %f\n", f_sensor);

	mindistL = 99999;
    mindistR = 99999;
    for (int i = 0; i < prey_loc.Size(); i++){
        double d = prey_loc[i] - pos;
        // printf("d = %f\n", d);
        if (d < 0 && d >= -HalfSpace){
            // Closest to the left side, distance is as calculated
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d < 0 && d < -HalfSpace){
            // Closest to the right side, distance is total area + - left side distance 
            d = SpaceSize + d;
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d > 0 && d > HalfSpace){ 
            // Closest to the left side, distance is -total area + right side distance
            d = -SpaceSize + d;
            if (abs(d) < mindistL){
                mindistL = abs(d);
            }
        }
        else if (d > 0 && d <= HalfSpace){
            // Closest to the right side, distance is as calculated
            if (d < mindistR){
                mindistR = d;
            }
        }
        else if (d == 0){
            d = 0;
        }
        else{
            printf("Prey pos size = %d\n", prey_loc.Size());
            printf("Error in predator prey sensing\n");
            printf("d = %f\n", d);
        }
    }
    // Cumulate, distance fits in the Gaussian as the difference of the mean (position) and the state (food position)
    // Negate left so negative sensor reading says food left, positive says food right, zero says no food or food on both sides.
	p_sensor = -2*exp(-(mindistL) * (mindistL) / (2 * s_width * s_width)) + 2*exp(-(mindistR) * (mindistR) / (2 * s_width * s_width));
    // printf("Pred sensor = %f\n", sensor);

    // Sense Hutch
    double distL = hutchL - pos;
    if (distL < 0){
        distL = SpaceSize + distL;
    }
    double distR = hutchR - pos;
    if (distR < 0){
        distR = SpaceSize + distR;
    }
    h_sensor = -2*exp(-(distL) * (distL) / (2 * s_width * s_width)) + 2*exp(-(distR) * (distR) / (2 * s_width * s_width));
}

// Step
void Predator::Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist, bool hutchflag, int hutchL, int hutchR)
{
    // Remember past position
    pastpos = pos;
	// Update the body position based on the other 2 neurons
    // If still handling previous catch, don't move
    if (handling == true){
        handling_counter += 1;
        if (handling_counter >= handling_time){
            handling = false;
            handling_counter = 0;
        }
    }
    else{
        double N1IP = f_sensor*sensorweights[1] + p_sensor*sensorweights[2] + h_sensor*sensorweights[3] + state*sensorweights[4];
        double N2IP = f_sensor*sensorweights[5] + p_sensor*sensorweights[6] + h_sensor*sensorweights[7] + state*sensorweights[8];
        double N3IP = f_sensor*sensorweights[9] + p_sensor*sensorweights[10] + h_sensor*sensorweights[11] + state*sensorweights[12];
        // Give each interneuron its sensory input
        NervousSystem.SetNeuronExternalInput(1, N1IP);
        NervousSystem.SetNeuronExternalInput(2, N2IP);
        NervousSystem.SetNeuronExternalInput(3, N3IP);
        // Update the nervous system
        NervousSystem.EulerStep(StepSize);
        // Check if trying to move into hutch
        pos += StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));
        mov_dist = StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));
        // if (hutchflag == false){
        //     if (pos > hutchL && pos < hutchR){
        //         pos = pastpos;
        //         mov_dist = 0.0;
        //     }
        // }
        // else{
        //     if (pos > hutchL || pos < hutchR){
        //         pos = pastpos;
        //         mov_dist = 0.0;
        //     }
        // }
        // Update State if the agent passed food
        if (pastpos < pos){
            if (pos > WorldFood.Size()){
                pos = pos - WorldFood.Size();
                if (hutchflag == false){
                    if (pos > hutchL && pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos > pastpos && preylist[i].pos <= WorldFood.Size()){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                            else if (preylist[i].pos >= 0 && preylist[i].pos < pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                        }
                    }
                }
                else{
                    if (pos > hutchL || pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos > pastpos && preylist[i].pos <= WorldFood.Size()){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                            else if (preylist[i].pos >= 0 && preylist[i].pos < pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                        }
                    }
                }   
            }
            else {
                if (hutchflag == false){
                    if (pos > hutchL && pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos > pastpos && preylist[i].pos <= pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                // handling = true;
                            }
                        }
                    }
                }
                else{
                    if (pos > hutchL || pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos > pastpos && preylist[i].pos <= pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                // handling = true;
                            }
                        }
                    }
                }
            }
        }
        if (pastpos > pos){
            if (pos < 0){
                pos = pos + WorldFood.Size();
                if (hutchflag == false){
                    if (pos > hutchL && pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos < pastpos && preylist[i].pos >= 0){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                            else if (preylist[i].pos <= WorldFood.Size() && preylist[i].pos > pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                        }
                    }
                }
                else{
                    if (pos > hutchL || pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos < pastpos && preylist[i].pos >= 0){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                            else if (preylist[i].pos <= WorldFood.Size() && preylist[i].pos > pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                handling = true;
                            }
                        }
                    }
                }
            }
            if (pos >= 0) {
                if (hutchflag == false){
                    if (pos > hutchL && pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos < pastpos && preylist[i].pos >= pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                // handling = true;
                            }
                        }
                    }
                }
                else{
                    if (pos > hutchL || pos < hutchR){
                    }
                    else{
                        for (int i = 0; i < preylist.Size(); i++){
                            if (preylist[i].pos < pastpos && preylist[i].pos >= pos){
                                state += preylist[i].state*feff;
                                fitness += preylist[i].state*feff;
                                preylist[i].state -= preylist[i].state*frate;
                                preylist[i].fitness -= preylist[i].state*frate;
                                snackflag += 1;
                                // handling = true;
                            }
                        }
                    }
                }
            }
        }
    }
    // Lose state over time
    // BASELINE 
    // state -= metaloss;
    // NCOT & PCOT (metaloss)
    if (mov_dist > NCOT_thresh){
        state -= metaloss + PCOT*metaloss + NCOT*mov_dist;
        fitness -= metaloss + PCOT*metaloss + NCOT*mov_dist;
    }
    else{
        state -= metaloss;
        fitness -= metaloss;
    }


    // Birth & Death
    if (state <= 0){
        death = true;
    }
    if (state > birth_thresh){
        birth = true;
    }
}

