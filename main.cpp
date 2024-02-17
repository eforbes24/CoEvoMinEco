// Eden Forbes
// MinCogEco Script

// ***************************************
// INCLUDES
// ***************************************

#include "Prey.h"
#include "Predator.h"
#include "random.h"
#include "TSearch.h"
#include <iostream>
#include <iomanip> 
#include <vector>
#include <string>
#include <list>

// ================================================
// A. PARAMETERS & GEN-PHEN MAPPING
// ================================================

// Run constants
// Make sure SpaceSize is also specified in the Prey.cpp and Predator.cpp files
const int SpaceSize = 5000;
const int HutchSize = 500;
const int CC = 2;
// 0-Indexed (0 = 1)
const int start_prey = 0;
const int start_pred = 0;


// Time Constants
// Evolution Step Size:
const double StepSize = 0.1;
// Analysis Step Size:
const double BTStepSize = 0.1;
// Evolution Run Time:
const double RunDuration = 5000;
// Behavioral Trace Run Time:
const double PlotDuration = 7500;
// EcoRate Collection Run Time:
const double RateDuration = 50000;
// Sensory Sample Run Time:
const double SenseDuration = 2000; // 2000

// EA params
const int PREY_POPSIZE = 50;
const int PRED_POPSIZE = 5;
const int GENS = 200;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;
// Number of trials per trial type (there are maxCC+1 * 2 trial types)
const double n_trials = 1.0;
// Fitness needed to switch in coevolution
// 0.0 means the agent broke even metabolically (survived on average)
const double swapper = 0.0;
const double swaps = 50;

// Nervous system params
const int prey_netsize = 3; 
const int pred_netsize = 3; 
const int num_prey_sensors = 4;
const int num_pred_sensors = 4;
// weight range 
const double WR = 16.0;
// sensor range
const double SR = 20.0;
// bias range
const double BR = 16.0;
// time constant min max
const double TMIN = 0.5;
const double TMAX = 20.0;
// Prey (Weights + TCs & Biases + SensorWeights + PhysiologicalParams) + Pred (Weights + TCs & Biases + SensorWeights + PhysiologicalParams)
const int PreyVectSize = (prey_netsize*prey_netsize + 2*prey_netsize + 4*prey_netsize);
const int PredVectSize = (pred_netsize*pred_netsize + 2*pred_netsize + 4*pred_netsize);

// Producer Parameters
const double G_Rate = 0.001*StepSize;
const double BT_G_Rate = 0.001*BTStepSize;

// Prey Sensory Parameters
const double prey_gain = 3.0;
const double prey_s_width = 100.0;

// Prey Metabolic Parameters
const double prey_loss_scalar = 4.0;
const double prey_frate = 0.15;
const double prey_feff = 0.1;
const double prey_repo = 1.5;
const double prey_b_thresh = 3.0;
const double prey_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*StepSize))) * prey_loss_scalar;
const double prey_BT_metaloss = ((prey_feff*(CC+1))/(SpaceSize/(prey_gain*BTStepSize))) * prey_loss_scalar;
const double prey_PCOT_scalar = 0.5;
const double prey_NCOT_scalar = 0.5;
const double prey_NCOT_thresh = 0.5;

// Predator Sensory Parameters 
const double pred_gain = 3.0;
const double pred_s_width = 100.0;

// Predator Metabolic Parameters
const double pred_loss_scalar = 2.0;
const double pred_frate = 1.0;
const double pred_feff = 0.9;
const double pred_repo = 2.5;
const double pred_b_thresh = 5.0;
const double pred_metaloss = (pred_feff/(SpaceSize/(pred_gain*StepSize))) * pred_loss_scalar;
const double pred_BT_metaloss = (pred_feff/(SpaceSize/(pred_gain*BTStepSize))) * pred_loss_scalar;
const double pred_handling_time = 10.0/StepSize;
const double pred_BT_handling_time = 10.0/BTStepSize;
const double pred_PCOT_scalar = 0.5;
const double pred_NCOT_scalar = 0.5;
const double pred_NCOT_thresh = 0.5;

// ------------------------------------
// Genotype-Phenotype Mapping Function
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, double species)
{
    double size = 0;
    if (gen.Size() == PreyVectSize){
        size = prey_netsize;
    }
    else{
        size = pred_netsize;
    }
	int k = 1;
	// Time-constants
	for (int i = 1; i <= size; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= size; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= size; i++) {
		for (int j = 1; j <= size; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
	// Sensor Weights
    if(species == 0){
        for (int i = 1; i <= size*num_prey_sensors; i++) {
            phen(k) = MapSearchParameter(gen(k), -SR, SR);
            k++;
        }
    }
    else{
        for (int i = 1; i <= size*num_pred_sensors; i++) {
            phen(k) = MapSearchParameter(gen(k), -SR, SR);
            k++;
        }
    }
}

void PhenNSMappingPrey(Prey &Agent1, TVector<double> &phen){
    int k = 1;
    // Prey Time-constants
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i,phen(k));
        k++;
    }
    // Prey Biases
    for (int i = 1; i <= prey_netsize; i++) {
        Agent1.NervousSystem.SetNeuronBias(i,phen(k));
        k++;
    }
    // Prey Neural Weights
    for (int i = 1; i <= prey_netsize; i++) {
        for (int j = 1; j <= prey_netsize; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i,j,phen(k));
            k++;
        }
    }
    // Prey Sensor Weights
    for (int i = 1; i <= prey_netsize*num_prey_sensors; i++) {
        Agent1.sensorweights[i] = phen(k);
        k++;
    }
}

void PhenNSMappingPred(Predator &Agent2, TVector<double> &phen){
    int k = 1;
    // Predator Time-constants
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronTimeConstant(i,phen(k));
        k++;
    }
    // Predator Biases
    for (int i = 1; i <= pred_netsize; i++) {
        Agent2.NervousSystem.SetNeuronBias(i,phen(k));
        k++;
    }
    // Predator Neural Weights
    for (int i = 1; i <= pred_netsize; i++) {
        for (int j = 1; j <= pred_netsize; j++) {
            Agent2.NervousSystem.SetConnectionWeight(i,j,phen(k));
            k++;
        }
    }
    // Predator Sensor Weights
    for (int i = 1; i <= pred_netsize*num_pred_sensors; i++) {
        Agent2.sensorweights[i] = phen(k);
        k++;
    }
}

// ================================================
// B. TASK ENVIRONMENT & FITNESS FUNCTION
// ================================================
double PreyTest(TVector<double> &genotype, RandomState &rs, TVector<double> &bestpred) 
{
    // Set running outcome variable
    // For Average Fitness
    double outcome = 0.0;
    // Translate genotype to phenotype
	TVector<double> preyphenotype;
	preyphenotype.SetBounds(1, PreyVectSize);
	GenPhenMapping(genotype, preyphenotype, 0);
    TVector<double> predphenotype;
	predphenotype.SetBounds(1, PredVectSize);
    GenPhenMapping(bestpred, predphenotype, 1);
    // Initialize Prey & Predator agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh, prey_PCOT_scalar, prey_NCOT_scalar, prey_NCOT_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time, pred_PCOT_scalar, pred_NCOT_scalar, pred_NCOT_thresh);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    PhenNSMappingPrey(Agent1, preyphenotype);
    PhenNSMappingPred(Agent2, predphenotype);

    // Set Trial Structure - Fixed CC
    double CoexistTrials = n_trials;
    // Run Simulation
    for (int trial = 0; trial < CoexistTrials; trial++){
        // Reset Prey agent, randomize its location and reset fitness
        Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        Agent1.fitness = 0.0;
        Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        Agent2.fitness = 0.0;
        // Seed preylist with starting population
        TVector<Prey> preylist(0,0);
        preylist[0] = Agent1;
        // Seed predlist with starting population
        TVector<Predator> predlist(0,0);
        predlist[0] = Agent2;
        // Initialize Producers
        TVector<double> food_pos;
        TVector<double> WorldFood(1, SpaceSize);
        WorldFood.FillContents(0.0);
        // Make Hutch
        bool hutchflag = false;
        int hutchL = rs.UniformRandomInteger(1,SpaceSize);
        int hutchR = hutchL + HutchSize;
        if (hutchR > SpaceSize){
            hutchR = hutchR - SpaceSize;
            hutchflag = true;
        }
        // Fill world to carrying capacity, with no food in the hutch
        for (int i = 0; i <= CC; i++){
            int f = rs.UniformRandomInteger(1,SpaceSize);
            if (hutchflag == false){
                if (f >= hutchL && f <= hutchR){
                    i--;
                    continue;
                }
                else{
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            else if (hutchflag == true){
                if (f >= hutchL || f <= hutchR){
                    i--;
                    continue;
                }
                else{
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        }
        // Set Clocks & trial outcome variables
        double clock = 0.0;
        // Run a Trial
        for (double time = 0; time < RunDuration; time += StepSize){
            // Remove any consumed food from food list
            TVector<double> dead_food(0,-1);
            for (int i = 0; i < food_pos.Size(); i++){
                if (WorldFood[food_pos[i]] <= 0){
                    dead_food.SetBounds(0, dead_food.Size());
                    dead_food[dead_food.Size()-1] = food_pos[i];
                }
            }
            if (dead_food.Size() > 0){
                for (int i = 0; i < dead_food.Size(); i++){
                    food_pos.RemoveFood(dead_food[i]);
                    food_pos.SetBounds(0, food_pos.Size()-2);
                }
            }
            // Chance for new food to grow
            // Carrying capacity is 0 indexed, add 1 for true amount
            for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    if (hutchflag == false){
                        if (f >= hutchL && f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    else if (hutchflag == true){
                        if (f >= hutchL || f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            // Update Prey Positions
            TVector<double> prey_pos;
            for (int i = 0; i < preylist.Size(); i++){
                prey_pos.SetBounds(0, prey_pos.Size());
                prey_pos[prey_pos.Size()-1] = preylist[i].pos;
            }
            // Predator Sense & Step
            for (int i = 0; i < predlist.Size(); i++){
                predlist[i].Sense(prey_pos, food_pos, hutchflag, hutchL, hutchR);
                predlist[i].Step(StepSize, WorldFood, preylist, hutchflag, hutchL, hutchR);
                if (predlist[i].birth == true){
                    predlist[i].state = predlist[i].state - prey_repo;
                    predlist[i].birth = false;
                }
                if (predlist[i].death == true){
                    predlist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
                    predlist[i].death = false;
                }
            }
            // Update Predator Positions
            TVector<double> pred_pos;
            for (int i = 0; i < predlist.Size(); i++){
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predlist[i].pos;
            }
            // Prey Sense & Step
            for (int i = 0; i < preylist.Size(); i++){
                preylist[i].Sense(food_pos, pred_pos, hutchflag, hutchL, hutchR);
                preylist[i].Step(StepSize, WorldFood);
                if (preylist[i].birth == true){
                    preylist[i].state = preylist[i].state - prey_repo;
                    preylist[i].birth = false;
                }
                if (preylist[i].death == true){
                    preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
                    preylist[i].death = false;
                }
            }
            // Update clocks
            clock += StepSize;
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
        }
        outcome += preylist[0].fitness;
    }
    double final_outcome = outcome/n_trials;
    return final_outcome;
}

double PredTest(TVector<double> &genotype, RandomState &rs, TVector<double> &bestprey) 
{
    // Set running outcome variable
    // For Average Fitness
    double outcome = 0.0;
    // Translate genotype to phenotype
	TVector<double> preyphenotype;
	preyphenotype.SetBounds(1, PreyVectSize);
	GenPhenMapping(bestprey, preyphenotype, 0);
    TVector<double> predphenotype;
	predphenotype.SetBounds(1, PredVectSize);
    GenPhenMapping(genotype, predphenotype, 1);
    // Initialize Prey & Predator agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_metaloss, prey_b_thresh, prey_PCOT_scalar, prey_NCOT_scalar, prey_NCOT_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_metaloss, pred_b_thresh, pred_handling_time, pred_PCOT_scalar, pred_NCOT_scalar, pred_NCOT_thresh);
    // Set Prey nervous system
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    PhenNSMappingPrey(Agent1, preyphenotype);
    PhenNSMappingPred(Agent2, predphenotype);
    // Set Trial Structure - Fixed CC
    double CoexistTrials = n_trials;
    // Run Simulation
    for (int trial = 0; trial < CoexistTrials; trial++){
        // Reset Prey agent, randomize its location. reset fitness
        Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
        Agent1.fitness = 0.0;
        Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
        Agent2.fitness = 0.0;
        // Seed preylist with starting population
        TVector<Prey> preylist(0,0);
        preylist[0] = Agent1;
        // Seed predlist with starting population
        TVector<Predator> predlist(0,0);
        predlist[0] = Agent2;
        // Initialize Producers, fill world to carrying capacity
        TVector<double> food_pos;
        TVector<double> WorldFood(1, SpaceSize);
        WorldFood.FillContents(0.0);
        // Make Hutch
        bool hutchflag = false;
        int hutchL = rs.UniformRandomInteger(1,SpaceSize);
        int hutchR = hutchL + HutchSize;
        if (hutchR > SpaceSize){
            hutchR = hutchR - SpaceSize;
            hutchflag = true;
        }
        // Fill world to carrying capacity, with no food in the hutch
        for (int i = 0; i <= CC; i++){
            int f = rs.UniformRandomInteger(1,SpaceSize);
            if (hutchflag == false){
                if (f >= hutchL && f <= hutchR){
                    i--;
                    continue;
                }
                else{
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            else if (hutchflag == true){
                if (f >= hutchL || f <= hutchR){
                    i--;
                    continue;
                }
                else{
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        }
        // Set Clocks & trial outcome variables
        double clock = 0.0;
        // Run a Trial
        for (double time = 0; time < RunDuration; time += StepSize){
            // Remove any consumed food from food list
            TVector<double> dead_food(0,-1);
            for (int i = 0; i < food_pos.Size(); i++){
                if (WorldFood[food_pos[i]] <= 0){
                    dead_food.SetBounds(0, dead_food.Size());
                    dead_food[dead_food.Size()-1] = food_pos[i];
                }
            }
            if (dead_food.Size() > 0){
                for (int i = 0; i < dead_food.Size(); i++){
                    food_pos.RemoveFood(dead_food[i]);
                    food_pos.SetBounds(0, food_pos.Size()-2);
                }
            }
            // Chance for new food to grow
            // Carrying capacity is 0 indexed, add 1 for true amount
            for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    if (hutchflag == false){
                        if (f >= hutchL && f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    else if (hutchflag == true){
                        if (f >= hutchL || f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
            // Update Prey Positions
            TVector<double> prey_pos;
            for (int i = 0; i < preylist.Size(); i++){
                prey_pos.SetBounds(0, prey_pos.Size());
                prey_pos[prey_pos.Size()-1] = preylist[i].pos;
            }
            // Predator Sense & Step
            for (int i = 0; i < predlist.Size(); i++){
                predlist[i].Sense(prey_pos, food_pos, hutchflag, hutchL, hutchR);
                predlist[i].Step(StepSize, WorldFood, preylist, hutchflag, hutchL, hutchR);
                if (predlist[i].birth == true){
                    predlist[i].state = predlist[i].state - prey_repo;
                    predlist[i].birth = false;
                }
                if (predlist[i].death == true){
                    predlist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
                    predlist[i].death = false;
                }
            }
            // Update Predator Positions
            TVector<double> pred_pos;
            for (int i = 0; i < predlist.Size(); i++){
                pred_pos.SetBounds(0, pred_pos.Size());
                pred_pos[pred_pos.Size()-1] = predlist[i].pos;
            }
            // Prey Sense & Step
            for (int i = 0; i < preylist.Size(); i++){
                preylist[i].Sense(food_pos, pred_pos, hutchflag, hutchL, hutchR);
                preylist[i].Step(StepSize, WorldFood);
                if (preylist[i].birth == true){
                    preylist[i].state = preylist[i].state - prey_repo;
                    preylist[i].birth = false;
                }
                if (preylist[i].death == true){
                    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
                    preylist[i].death = false;
                }
            }
            // Update clocks
            clock += StepSize;
            prey_pos.~TVector();
            pred_pos.~TVector();
            dead_food.~TVector();
            
        }
        // // Take average fitness value across trials
        outcome += predlist[0].fitness;
    }
    double final_outcome = outcome/n_trials;
    // printf("Predator Outcome: %f\n", final_outcome);
    return final_outcome;
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int IntTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= swapper) return 1;
	else return 0;
}

int EndTerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf >= 100.0) return 1;
	else return 0;
}

void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
    BestIndividualFile << setprecision(32);
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ================================================
// D. ANALYSIS FUNCTIONS
// ================================================

// // ------------------------------------
// // Interaction Rate Data Functions
// // ------------------------------------
// void DeriveLambdaH(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH.dat");
//     // for (int i = 0; i <= 0; i++){
//         TVector<TVector<double> > lambH;
//         for (int j = 0; j <= maxCC; j++){
//             TVector<double> lambHcc;
//             for (int k = 0; k <= samplesize; k++){
//                 int carrycapacity = j;
//                 // Fill World to Carrying Capacity
//                 TVector<double> food_pos;
//                 TVector<double> WorldFood(1, SpaceSize);
//                 WorldFood.FillContents(0.0);
//                 for (int i = 0; i <= carrycapacity; i++){
//                     int f = rs.UniformRandomInteger(1,SpaceSize);
//                     WorldFood[f] = 1.0;
//                     food_pos.SetBounds(0, food_pos.Size());
//                     food_pos[food_pos.Size()-1] = f;
//                 }
//                 // Seed preylist with starting population
//                 TVector<Prey> preylist(0,0);
//                 TVector<double> prey_pos;
//                 preylist[0] = prey;
//                 // for (int i = 0; i < j; i++){
//                 //     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                 //     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                 //     newprey.NervousSystem = prey.NervousSystem;
//                 //     newprey.sensorweights = prey.sensorweights;
//                 //     preylist.SetBounds(0, preylist.Size());
//                 //     preylist[preylist.Size()-1] = newprey;
//                 //     }
//                 // Make dummy predator list
//                 TVector<double> pred_pos(0,-1);
//                 double munch_count = 0;
//                 for (double time = 0; time < RateDuration; time += BTStepSize){
//                     // Remove chomped food from food list
//                     TVector<double> dead_food(0,-1);
//                     for (int i = 0; i < food_pos.Size(); i++){
//                         if (WorldFood[food_pos[i]] <= 0){
//                             dead_food.SetBounds(0, dead_food.Size());
//                             dead_food[dead_food.Size()-1] = food_pos[i];
//                         }
//                     }
//                     if (dead_food.Size() > 0){
//                         for (int i = 0; i < dead_food.Size(); i++){
//                             food_pos.RemoveFood(dead_food[i]);
//                             food_pos.SetBounds(0, food_pos.Size()-2);
//                         }
//                     }
//                     // Carrying capacity is 0 indexed, add 1 for true amount
//                     for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
//                         double c = rs.UniformRandom(0,1);
//                         if (c <= BT_G_Rate){
//                             int f = rs.UniformRandomInteger(1,SpaceSize);
//                             WorldFood[f] = 1.0;
//                             food_pos.SetBounds(0, food_pos.Size());
//                             food_pos[food_pos.Size()-1] = f;
//                         }
//                     }
//                     for (int i = 0; i < preylist.Size(); i++){
//                         // Prey Sense & Step
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                         // Check Births
//                         if (preylist[i].birth == true){
//                             preylist[i].state = preylist[i].state - prey_repo;
//                             preylist[i].birth = false;
//                         }
//                         // Check Deaths
//                         if (preylist[i].death == true){
//                             preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
//                             preylist[i].death = false;
//                         }
//                         // Check # of times food crossed
//                         if (time > transient){
//                             munch_count += preylist[i].snackflag;
//                             preylist[i].snackflag = 0.0;
//                         }
//                     }
//                 }
//                 double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//                 lambHcc.SetBounds(0, lambHcc.Size());
//                 lambHcc[lambHcc.Size()-1] = munchrate;
//             }
//         //     lambH.SetBounds(0, lambH.Size());
//         //     lambH[lambH.Size()-1] = lambHcc;
//         // }
//         lambHfile << lambHcc << endl;
//         lambHcc.~TVector();
//     }
//     // Save
//     lambHfile.close();
// }

// void DeriveLambdaH2(Prey &prey, Predator &predator, RandomState &rs, double &maxCC, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambHfile("menagerie/IndBatch2/analysis_results/ns_15/lambH3.dat");
//     // for (int i = 0; i <= 0; i++){
//         TVector<TVector<double> > lambH;
//         for (int j = -1; j <= maxprey; j++){
//             TVector<double> lambHcc;
//             for (int k = 0; k <= samplesize; k++){
//                 int carrycapacity = 29;
//                 // Fill World to Carrying Capacity
//                 TVector<double> food_pos;
//                 TVector<double> WorldFood(1, SpaceSize);
//                 WorldFood.FillContents(0.0);
//                 for (int i = 0; i <= carrycapacity; i++){
//                     int f = rs.UniformRandomInteger(1,SpaceSize);
//                     WorldFood[f] = 1.0;
//                     food_pos.SetBounds(0, food_pos.Size());
//                     food_pos[food_pos.Size()-1] = f;
//                 }
//                 // Seed preylist with starting population
//                 TVector<Prey> preylist(0,0);
//                 TVector<double> prey_pos;
//                 preylist[0] = prey;
//                 for (int i = 0; i < j; i++){
//                     Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                     newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                     newprey.NervousSystem = prey.NervousSystem;
//                     newprey.sensorweights = prey.sensorweights;
//                     preylist.SetBounds(0, preylist.Size());
//                     preylist[preylist.Size()-1] = newprey;
//                     }
//                 // Make dummy predator list
//                 TVector<double> pred_pos(0,-1);
//                 double munch_count = 0;
//                 for (double time = 0; time < RateDuration; time += BTStepSize){
//                     // Remove chomped food from food list
//                     TVector<double> dead_food(0,-1);
//                     for (int i = 0; i < food_pos.Size(); i++){
//                         if (WorldFood[food_pos[i]] <= 0){
//                             dead_food.SetBounds(0, dead_food.Size());
//                             dead_food[dead_food.Size()-1] = food_pos[i];
//                         }
//                     }
//                     if (dead_food.Size() > 0){
//                         for (int i = 0; i < dead_food.Size(); i++){
//                             food_pos.RemoveFood(dead_food[i]);
//                             food_pos.SetBounds(0, food_pos.Size()-2);
//                         }
//                     }
//                     // Carrying capacity is 0 indexed, add 1 for true amount
//                     for (int i = 0; i < ((carrycapacity+1) - food_pos.Size()); i++){
//                         double c = rs.UniformRandom(0,1);
//                         if (c <= BT_G_Rate){
//                             int f = rs.UniformRandomInteger(1,SpaceSize);
//                             WorldFood[f] = 1.0;
//                             food_pos.SetBounds(0, food_pos.Size());
//                             food_pos[food_pos.Size()-1] = f;
//                         }
//                     }
//                     for (int i = 0; i < preylist.Size(); i++){
//                         // Prey Sense & Step
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                         // Check Births
//                         if (preylist[i].birth == true){
//                             preylist[i].state = preylist[i].state - prey_repo;
//                             preylist[i].birth = false;
//                         }
//                         // Check Deaths
//                         if (preylist[i].death == true){
//                             preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 2.0);
//                             preylist[i].death = false;
//                         }
//                         // Check # of times food crossed
//                         if (time > transient){
//                             munch_count += preylist[i].snackflag;
//                             preylist[i].snackflag = 0.0;
//                         }
//                     }
//                 }
//                 double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//                 lambHcc.SetBounds(0, lambHcc.Size());
//                 lambHcc[lambHcc.Size()-1] = munchrate;
//             }
//         //     lambH.SetBounds(0, lambH.Size());
//         //     lambH[lambH.Size()-1] = lambHcc;
//         // }
//         lambHfile << lambHcc << endl;
//         lambHcc.~TVector();
//     }
//     // Save
//     lambHfile.close();
// }

// void DeriveRR(RandomState &rs, double &testCC, int &samplesize)
// {
//     ofstream RRfile("menagerie/IndBatch2/analysis_results/ns_15/RR.dat");
//     for (int r = -1; r <= testCC; r++){
//         TVector<double> RR;
//         for (int k = 0; k <= samplesize; k++){
//             double counter = 0;
//             for (double time = 0; time < RateDuration; time += BTStepSize){
//                 for (int i = 0; i < ((testCC+1) - (r+1)); i++){
//                     double c = rs.UniformRandom(0,1);
//                     if (c <= BT_G_Rate){
//                         int f = rs.UniformRandomInteger(1,SpaceSize);
//                         counter += 1;
//                     }
//                 }
//             }
//             RR.SetBounds(0, RR.Size());
//             RR[RR.Size()-1] = counter/(RateDuration/BTStepSize);
//         }
//         RRfile << RR << endl;
//         RR.~TVector();
//     }
//     // Save
//     RRfile.close();
// }

// void DeriveLambdaP(Prey &prey, Predator &predator, RandomState &rs, double &maxprey, int &samplesize, double &transient)
// {
//     ofstream lambCfile("menagerie/IndBatch2/analysis_results/ns_15/lambC.dat");
//     for (int j = -1; j<=maxprey; j++)
//     {   
//         TVector<double> lambC;
//         for (int k = 0; k<=samplesize; k++){
//             // Fill World to Carrying Capacity
//             TVector<double> food_pos;
//             TVector<double> WorldFood(1, SpaceSize);
//             WorldFood.FillContents(0.0);
//             for (int i = 0; i <= CC; i++){
//                 int f = rs.UniformRandomInteger(1,SpaceSize);
//                 WorldFood[f] = 1.0;
//                 food_pos.SetBounds(0, food_pos.Size());
//                 food_pos[food_pos.Size()-1] = f;
//             }
//             // Seed preylist with starting population
//             TVector<Prey> preylist(0,0);
//             TVector<double> prey_pos;
//             preylist[0] = prey;
//             for (int i = 0; i < j; i++){
//                 Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//                 newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                 newprey.NervousSystem = prey.NervousSystem;
//                 newprey.sensorweights = prey.sensorweights;
//                 preylist.SetBounds(0, preylist.Size());
//                 preylist[preylist.Size()-1] = newprey;
//                 }
//             double munch_count = 0;
//             for (double time = 0; time < RateDuration; time += BTStepSize){
//                 // Remove chomped food from food list
//                 TVector<double> dead_food(0,-1);
//                 for (int i = 0; i < food_pos.Size(); i++){
//                     if (WorldFood[food_pos[i]] <= 0){
//                         dead_food.SetBounds(0, dead_food.Size());
//                         dead_food[dead_food.Size()-1] = food_pos[i];
//                     }
//                 }
//                 if (dead_food.Size() > 0){
//                     for (int i = 0; i < dead_food.Size(); i++){
//                         food_pos.RemoveFood(dead_food[i]);
//                         food_pos.SetBounds(0, food_pos.Size()-2);
//                     }
//                 }
//                 // Carrying capacity is 0 indexed, add 1 for true amount
//                 for (int i = 0; i < ((CC+1) - food_pos.Size()); i++){
//                     double c = rs.UniformRandom(0,1);
//                     if (c <= BT_G_Rate){
//                         int f = rs.UniformRandomInteger(1,SpaceSize);
//                         WorldFood[f] = 1.0;
//                         food_pos.SetBounds(0, food_pos.Size());
//                         food_pos[food_pos.Size()-1] = f;
//                     }
//                 }
//                 // Prey Sense & Step
//                 TVector<Prey> newpreylist;
//                 TVector<int> deaths;
//                 TVector<double> prey_pos;
//                 TVector<double> pred_pos;
//                 pred_pos.SetBounds(0, pred_pos.Size());
//                 pred_pos[pred_pos.Size()-1] = predator.pos;
//                 for (int i = 0; i < preylist.Size(); i++){
//                     if (preylist[i].death == true){
//                         preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//                     }
//                     else{
//                         preylist[i].Sense(food_pos, pred_pos);
//                         preylist[i].Step(BTStepSize, WorldFood);
//                     }
//                 }
//                 for (int i = 0; i <= preylist.Size()-1; i++){
//                     prey_pos.SetBounds(0, prey_pos.Size());
//                     prey_pos[prey_pos.Size()-1] = preylist[i].pos;
//                 }

//                 // Predator Sense & Step
//                 predator.Sense(prey_pos);
//                 predator.Step(BTStepSize, WorldFood, preylist);
//                 // Check # of times food crossed
//                 if(time > transient){
//                     munch_count += predator.snackflag;
//                     predator.snackflag = 0.0;
//                 }
//             }

//             double munchrate = munch_count/((RateDuration-transient)/BTStepSize);
//             lambC.SetBounds(0, lambC.Size());
//             lambC[lambC.Size()-1] = munchrate;
//         }
//         lambCfile << lambC << endl;
//         lambC.~TVector();
//     }
//     // Save
//     lambCfile.close();
// }

// // void CollectEcoRates(TVector<double> &genotype, RandomState &rs)
// {
//     ofstream erates("menagerie/IndBatch2/analysis_results/ns_15/ecosystem_rates.dat");
//     // Translate to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);
//     // Create agents
//     Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//     Predator Agent2(pred_gain, pred_s_width, pred_frate, pred_BT_handling_time);
//     Agent2.condition = pred_condition;
//     // Set nervous system
//     Agent1.NervousSystem.SetCircuitSize(prey_netsize);
//     int k = 1;
//     // Prey Time-constants
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
//         k++;
//     }
//     // Prey Biases
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
//         k++;
//     }
//     // Prey Neural Weights
//     for (int i = 1; i <= prey_netsize; i++) {
//         for (int j = 1; j <= prey_netsize; j++) {
//             Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
//             k++;
//         }
//     }
//     // Prey Sensor Weights
//     for (int i = 1; i <= prey_netsize*3; i++) {
//         Agent1.sensorweights[i] = phenotype(k);
//         k++;
//     }
//     // Save Growth Rates
//     // Max growth rate of producers is the chance of a new plant coming in on a given time step
//     double systemcc = CC+1; // 0 indexed
//     double rr = BT_G_Rate;
//     erates << rr << " ";
//     erates << systemcc << " ";
//     erates << Agent1.frate << " ";
//     erates << Agent1.feff << " ";
//     erates << Agent1.metaloss << " ";
//     erates << Agent2.frate << " ";
//     // erates << Agent2.feff << " ";
//     // erates << Agent2.metaloss << " ";
//     erates.close();

//     // Set Sampling Range & Frequency
//     double maxCC = 300;
//     double maxprey = 60;
//     double transient = 100.0;
//     int samplesize = 10;
//     double testCC = 29;
//     // Collect rr at testCC
//     printf("Collecting Growth Rate at Test Carrying Capacity\n");
//     DeriveRR(rs, testCC, samplesize);
//     // Collect Prey Lambda & r
//     printf("Collecting Prey rates\n");
//     DeriveLambdaH(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
//     DeriveLambdaH2(Agent1, Agent2, rs, maxCC, maxprey, samplesize, transient);
//     // Collect Predator Lambda & r
//     // printf("Collecting Predator rates\n");
//     // DeriveLambdaP(Agent1, Agent2, rs, maxprey, samplesize, transient);
// }

// ------------------------------------
// Sensory Sample Functions
// ------------------------------------
double SSCoexist(TVector<double> &prey_genotype, TVector<double> &pred_genotype, RandomState &rs, int agent, string batch) 
{
    // Start output files
    std::string PreySSfile("menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/Prey_SenS.dat");
    std::string PredSSfile("menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/Pred_SenS.dat");
    std::string pyfile = "menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/prey_pos.dat";
    std::string pdfile = "menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/pred_pos.dat";
    std::string ffile = "menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/food_pos.dat";
    std::string fpfile = "menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/food_pop.dat";
    std::string hfile = "menagerie/" + batch + "/analysis_results/ns_" + std::to_string(agent) + "/hutch.dat";
    ofstream preySS(PreySSfile);
    ofstream predSS(PredSSfile);
    ofstream preyfile(pyfile);
    ofstream predfile(pdfile);
    ofstream foodfile(ffile);
    ofstream foodpopfile(fpfile);
    ofstream hutchfile(hfile);
    // Set prey data collection vectors 
    TVector<double> prey_FS;
    TVector<double> prey_PS;
    TVector<double> prey_HS;
    TVector<double> prey_SS;
    TVector<double> prey_NO1;
    TVector<double> prey_N1FS;
    TVector<double> prey_N1PS;
    TVector<double> prey_N1HS;
    TVector<double> prey_N1SS;
    TVector<double> prey_NO2;
    TVector<double> prey_N2FS;
    TVector<double> prey_N2PS;
    TVector<double> prey_N2HS;
    TVector<double> prey_N2SS;
    TVector<double> prey_NO3;
    TVector<double> prey_N3FS;
    TVector<double> prey_N3PS;
    TVector<double> prey_N3HS;
    TVector<double> prey_N3SS;
    TVector<double> prey_mov;
    // Set predator data collection vectors
    TVector<double> pred_PS;
    TVector<double> pred_FS;
    TVector<double> pred_HS;
    TVector<double> pred_SS;
    TVector<double> pred_NO1;
    TVector<double> pred_N1FS;
    TVector<double> pred_N1PS;
    TVector<double> pred_N1HS;
    TVector<double> pred_N1SS;
    TVector<double> pred_NO2;
    TVector<double> pred_N2FS;
    TVector<double> pred_N2PS;
    TVector<double> pred_N2HS;
    TVector<double> pred_N2SS;
    TVector<double> pred_NO3;
    TVector<double> pred_N3FS;
    TVector<double> pred_N3PS;
    TVector<double> pred_N3HS;
    TVector<double> pred_N3SS;
    TVector<double> pred_mov;

    // Set running outcome
    double outcome = 99999999999.0;
    // Translate to phenotypes
	TVector<double> prey_phenotype;
	TVector<double> pred_phenotype;
	prey_phenotype.SetBounds(1, PreyVectSize);
    pred_phenotype.SetBounds(1, PredVectSize);
	GenPhenMapping(prey_genotype, prey_phenotype, 0);
	GenPhenMapping(pred_genotype, pred_phenotype, 1);
    // Create agents
    Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh, prey_PCOT_scalar, prey_NCOT_scalar, prey_NCOT_thresh);
    Predator Agent2(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time, pred_PCOT_scalar, pred_NCOT_scalar, pred_NCOT_thresh);
    // Set Prey nervous systems
    Agent1.NervousSystem.SetCircuitSize(prey_netsize);
    Agent2.NervousSystem.SetCircuitSize(pred_netsize);
    PhenNSMappingPrey(Agent1, prey_phenotype);
    PhenNSMappingPred(Agent2, pred_phenotype);

    // Run Simulation
    // Reset Agents & Vectors
    Agent1.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
    Agent2.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
    // Seed preylist with starting population
    TVector<Prey> preylist(0,0);
    preylist[0] = Agent1;
    for (int i = 0; i < start_prey; i++){
        Prey newprey(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh, prey_PCOT_scalar, prey_NCOT_scalar, prey_NCOT_thresh);
        newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
        newprey.NervousSystem = Agent1.NervousSystem;
        newprey.sensorweights = Agent1.sensorweights;
        preylist.SetBounds(0, preylist.Size());
        preylist[preylist.Size()-1] = newprey;
    }
    // Seed predlist with starting population
    TVector<Predator> predlist(0,0);
    predlist[0] = Agent2;
    for (int i = 0; i < start_pred; i++){
        Predator newpred(pred_netsize, pred_gain, pred_s_width, pred_frate, pred_feff, pred_BT_metaloss, pred_b_thresh, pred_BT_handling_time, pred_PCOT_scalar, pred_NCOT_scalar, pred_NCOT_thresh);
        newpred.Reset(rs.UniformRandomInteger(0,SpaceSize), 2.5);
        newpred.NervousSystem = Agent2.NervousSystem;
        newpred.sensorweights = Agent2.sensorweights;
        predlist.SetBounds(0, predlist.Size());
        predlist[predlist.Size()-1] = newpred;
    }
    // Fill World to Carrying Capacity
    TVector<double> food_pos(0,-1);
    TVector<double> WorldFood(1, SpaceSize);
    WorldFood.FillContents(0.0);
    // Make Hutch
    bool hutchflag = false;
    int hutchL = rs.UniformRandomInteger(1,SpaceSize);
    int hutchR = hutchL + HutchSize;
    if (hutchR > SpaceSize){
        hutchR = hutchR - SpaceSize;
        hutchflag = true;
    }

    // Fill world to carrying capacity, with no food in the hutch
    for (int i = 0; i <= CC; i++){
        int f = rs.UniformRandomInteger(1,SpaceSize);
        if (hutchflag == false){
            if (f >= hutchL && f <= hutchR){
                i--;
                continue;
            }
            else{
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
        }
        else if (hutchflag == true){
            if (f >= hutchL || f <= hutchR){
                i--;
                continue;
            }
            else{
                WorldFood[f] = 1.0;
                food_pos.SetBounds(0, food_pos.Size());
                food_pos[food_pos.Size()-1] = f;
            }
        }
    }
    // Run Simulation
    for (double time = 0; time < PlotDuration; time += BTStepSize){
        // Remove chomped food from food list
        TVector<double> dead_food(0,-1);
        for (int i = 0; i < food_pos.Size(); i++){
            if (WorldFood[food_pos[i]] <= 0){
                dead_food.SetBounds(0, dead_food.Size());
                dead_food[dead_food.Size()-1] = food_pos[i];
            }
        }
        if (dead_food.Size() > 0){
            for (int i = 0; i < dead_food.Size(); i++){
                food_pos.RemoveFood(dead_food[i]);
                food_pos.SetBounds(0, food_pos.Size()-2);
            }
        }
        // Carrying capacity is 0 indexed, add 1 for true amount
        for (int i = 0; i < CC+1 - food_pos.Size(); i++){
                double c = rs.UniformRandom(0,1);
                if (c <= G_Rate){
                    int f = rs.UniformRandomInteger(1,SpaceSize);
                    if (hutchflag == false){
                        if (f >= hutchL && f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    else if (hutchflag == true){
                        if (f >= hutchL || f <= hutchR){
                            f = f + HutchSize;
                            if (f > SpaceSize){
                                f = f - SpaceSize;
                            }
                        }
                    }
                    WorldFood[f] = 1.0;
                    food_pos.SetBounds(0, food_pos.Size());
                    food_pos[food_pos.Size()-1] = f;
                }
            }
        // Update Prey Positions
        TVector<double> prey_pos;
        for (int i = 0; i < preylist.Size(); i++){
            prey_pos.SetBounds(0, prey_pos.Size());
            prey_pos[prey_pos.Size()-1] = preylist[i].pos;
        }
        // Predator Sense & Step
        TVector<Predator> newpredlist;
        TVector<int> preddeaths;
        for (int i = 0; i < predlist.Size(); i++){
            predlist[i].Sense(prey_pos, food_pos, hutchflag, hutchL, hutchR);
            pred_PS.SetBounds(0, pred_PS.Size());
            pred_PS[pred_PS.Size()-1] = predlist[i].p_sensor;
            pred_N1PS.SetBounds(0, pred_N1PS.Size());
            pred_N1PS[pred_N1PS.Size()-1] = predlist[i].p_sensor * predlist[i].sensorweights[1];
            pred_N2PS.SetBounds(0, pred_N2PS.Size());
            pred_N2PS[pred_N2PS.Size()-1] = predlist[i].p_sensor * predlist[i].sensorweights[3];
            pred_N3PS.SetBounds(0, pred_N3PS.Size());
            pred_N3PS[pred_N3PS.Size()-1] = predlist[i].p_sensor * predlist[i].sensorweights[5];

            pred_FS.SetBounds(0, pred_FS.Size());
            pred_FS[pred_FS.Size()-1] = predlist[i].f_sensor;
            pred_N1FS.SetBounds(0, pred_N1FS.Size());
            pred_N1FS[pred_N1FS.Size()-1] = predlist[i].f_sensor * predlist[i].sensorweights[2];
            pred_N2FS.SetBounds(0, pred_N2FS.Size());
            pred_N2FS[pred_N2FS.Size()-1] = predlist[i].f_sensor * predlist[i].sensorweights[4];
            pred_N3FS.SetBounds(0, pred_N3FS.Size());
            pred_N3FS[pred_N3FS.Size()-1] = predlist[i].f_sensor * predlist[i].sensorweights[6];

            pred_HS.SetBounds(0, pred_HS.Size());
            pred_HS[pred_HS.Size()-1] = predlist[i].h_sensor;
            pred_N1HS.SetBounds(0, pred_N1HS.Size());
            pred_N1HS[pred_N1HS.Size()-1] = predlist[i].h_sensor * predlist[i].sensorweights[2];
            pred_N2HS.SetBounds(0, pred_N2HS.Size());
            pred_N2HS[pred_N2HS.Size()-1] = predlist[i].h_sensor * predlist[i].sensorweights[4];
            pred_N3HS.SetBounds(0, pred_N3HS.Size());
            pred_N3HS[pred_N3HS.Size()-1] = predlist[i].h_sensor * predlist[i].sensorweights[6];

            pred_SS.SetBounds(0, pred_SS.Size());
            pred_SS[pred_SS.Size()-1] = predlist[i].state;
            pred_N1SS.SetBounds(0, pred_N1SS.Size());
            pred_N1SS[pred_N1SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[2];
            pred_N2SS.SetBounds(0, pred_N2SS.Size());
            pred_N2SS[pred_N2SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[4];
            pred_N3SS.SetBounds(0, pred_N3SS.Size());
            pred_N3SS[pred_N3SS.Size()-1] = predlist[i].state * predlist[i].sensorweights[6];

            predlist[i].Step(BTStepSize, WorldFood, preylist, hutchflag, hutchL, hutchR);
            pred_NO1.SetBounds(0, pred_NO1.Size());
            pred_NO1[pred_NO1.Size()-1] = predlist[i].NervousSystem.NeuronOutput(1);
            pred_NO2.SetBounds(0, pred_NO2.Size());
            pred_NO2[pred_NO2.Size()-1] = predlist[i].NervousSystem.NeuronOutput(2);
            pred_NO3.SetBounds(0, pred_NO3.Size());
            pred_NO3[pred_NO3.Size()-1] = predlist[i].NervousSystem.NeuronOutput(3);
            pred_mov.SetBounds(0, pred_mov.Size());
            pred_mov[pred_mov.Size()-1] = (predlist[i].NervousSystem.NeuronOutput(2) - predlist[i].NervousSystem.NeuronOutput(1));
            
            if (predlist[i].birth == true){
                predlist[i].state = predlist[i].state - pred_repo;
                predlist[i].birth = false;
            }
            if (predlist[i].death == true){
                predlist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 5.0);
                predlist[i].death = false;
            }
        }
        // Update Predator Positions
        TVector<double> pred_pos;
        for (int i = 0; i < predlist.Size(); i++){
            pred_pos.SetBounds(0, pred_pos.Size());
            pred_pos[pred_pos.Size()-1] = predlist[i].pos;
        }
        // Prey Sense & Step
        TVector<Prey> newpreylist;
        TVector<int> preydeaths;
        for (int i = 0; i < preylist.Size(); i++){
            preylist[i].Sense(food_pos, pred_pos, hutchflag, hutchL, hutchR);
            prey_FS.SetBounds(0, prey_FS.Size());
            prey_FS[prey_FS.Size()-1] = preylist[i].f_sensor;
            prey_N1FS.SetBounds(0, prey_N1FS.Size());
            prey_N1FS[prey_N1FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[1];
            prey_N2FS.SetBounds(0, prey_N2FS.Size());
            prey_N2FS[prey_N2FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[4];
            prey_N3FS.SetBounds(0, prey_N3FS.Size());
            prey_N3FS[prey_N3FS.Size()-1] = preylist[i].f_sensor * preylist[i].sensorweights[7];
            prey_PS.SetBounds(0, prey_PS.Size());
            prey_PS[prey_PS.Size()-1] = preylist[i].p_sensor;
            prey_N1PS.SetBounds(0, prey_N1PS.Size());
            prey_N1PS[prey_N1PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[2];
            prey_N2PS.SetBounds(0, prey_N2PS.Size());
            prey_N2PS[prey_N2PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[5];
            prey_N3PS.SetBounds(0, prey_N3PS.Size());
            prey_N3PS[prey_N3PS.Size()-1] = preylist[i].p_sensor * preylist[i].sensorweights[8];
            prey_HS.SetBounds(0, prey_HS.Size());
            prey_HS[prey_HS.Size()-1] = preylist[i].h_sensor;
            prey_N1HS.SetBounds(0, prey_N1HS.Size());
            prey_N1HS[prey_N1HS.Size()-1] = preylist[i].h_sensor * preylist[i].sensorweights[2];
            prey_N2HS.SetBounds(0, prey_N2HS.Size());
            prey_N2HS[prey_N2HS.Size()-1] = preylist[i].h_sensor * preylist[i].sensorweights[5];
            prey_N3HS.SetBounds(0, prey_N3HS.Size());
            prey_N3HS[prey_N3HS.Size()-1] = preylist[i].h_sensor * preylist[i].sensorweights[8];
            prey_SS.SetBounds(0, prey_SS.Size());
            prey_SS[prey_SS.Size()-1] = preylist[i].state;
            prey_N1SS.SetBounds(0, prey_N1SS.Size());
            prey_N1SS[prey_N1SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[3];
            prey_N2SS.SetBounds(0, prey_N2SS.Size());
            prey_N2SS[prey_N2SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[6];
            prey_N3SS.SetBounds(0, prey_N3SS.Size());
            prey_N3SS[prey_N3SS.Size()-1] = preylist[i].state * preylist[i].sensorweights[9];

            preylist[i].Step(BTStepSize, WorldFood);
            prey_NO1.SetBounds(0, prey_NO1.Size());
            prey_NO1[prey_NO1.Size()-1] = preylist[i].NervousSystem.NeuronOutput(1);
            prey_NO2.SetBounds(0, prey_NO2.Size());
            prey_NO2[prey_NO2.Size()-1] = preylist[i].NervousSystem.NeuronOutput(2);
            prey_NO3.SetBounds(0, prey_NO3.Size());
            prey_NO3[prey_NO3.Size()-1] = preylist[i].NervousSystem.NeuronOutput(3);
            prey_mov.SetBounds(0, prey_mov.Size());
            prey_mov[prey_mov.Size()-1] = (preylist[i].NervousSystem.NeuronOutput(2) - preylist[i].NervousSystem.NeuronOutput(1));
            
            if (preylist[i].birth == true){
                preylist[i].state = preylist[i].state - prey_repo;
                preylist[i].birth = false;
            }
            if (preylist[i].death == true){
                preylist[i].Reset(rs.UniformRandomInteger(0,SpaceSize), 3.0);
                preylist[i].death = false;
            }
        }
        // Save
        preyfile << prey_pos << endl;
        predfile << pred_pos << endl;
        foodfile << food_pos << endl;
        double foodsum = 0.0;
        for (int i = 0; i < food_pos.Size(); i++){
            foodsum += WorldFood[food_pos[i]];
        }
        foodpopfile << foodsum << " ";
        // Check Population Collapse
        newpreylist.~TVector();
        preydeaths.~TVector();
        newpredlist.~TVector();
        preddeaths.~TVector();
        prey_pos.~TVector();
        pred_pos.~TVector();
        dead_food.~TVector();
    }
    preyfile.close();
    predfile.close();
	foodfile.close();
    foodpopfile.close();

    hutchfile << hutchL << " " << hutchR << " " << hutchflag << " ";
    hutchfile.close();

    preySS << prey_FS << endl << prey_N1FS << endl << prey_N2FS << endl << prey_N3FS << endl;
    preySS << prey_PS << endl << prey_N1PS << endl << prey_N2PS << endl << prey_N3PS << endl;
    preySS << prey_HS << endl << prey_N1HS << endl << prey_N2HS << endl << prey_N3HS << endl;
    preySS << prey_SS << endl << prey_N1SS << endl << prey_N2SS << endl << prey_N3SS << endl; 
    preySS << prey_NO1 << endl << prey_NO2 << endl << prey_NO3 << endl << prey_mov << endl;

    predSS << pred_PS << endl << pred_N1PS << endl << pred_N2PS << endl << pred_N3PS << endl;
    predSS << pred_FS << endl << pred_N1FS << endl << pred_N2FS << endl << pred_N3FS << endl;
    predSS << pred_HS << endl << pred_N1HS << endl << pred_N2HS << endl << pred_N3HS << endl;
    predSS << pred_SS << endl << pred_N1SS << endl << pred_N2SS << endl << pred_N3SS << endl; 
    predSS << pred_NO1 << endl << pred_NO2 << endl << pred_NO3 << endl << pred_mov << endl;

    preySS.close();
    predSS.close();
    
    return 0;
}

// // ---------------------------------------
// // EcoSim Sample Collection
// // ---------------------------------------
// void NewEco(TVector<double> &genotype, RandomState &rs)
// {
//     double test_CC = 29;
//     double start_CC = 10;
//     double start_prey_sim = 15;
//     double test_frate = prey_frate;
//     double test_feff = prey_feff;
//     ofstream ppfile("menagerie/IndBatch2/analysis_results/ns_15/sim_prey_pop.dat");
//     ofstream fpfile("menagerie/IndBatch2/analysis_results/ns_15/sim_food_pop.dat");
//     // Translate to phenotype
// 	TVector<double> phenotype;
// 	phenotype.SetBounds(1, VectSize);
// 	GenPhenMapping(genotype, phenotype);
//     // Create agents
//     // Playing with feff, frate, metaloss
//     Prey Agent1(prey_netsize, prey_gain, prey_s_width, prey_frate, prey_feff, prey_BT_metaloss, prey_b_thresh);
//     // Set nervous system
//     Agent1.NervousSystem.SetCircuitSize(prey_netsize);
//     int k = 1;
//     // Prey Time-constants
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
//         k++;
//     }
//     // Prey Biases
//     for (int i = 1; i <= prey_netsize; i++) {
//         Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
//         k++;
//     }
//     // Prey Neural Weights
//     for (int i = 1; i <= prey_netsize; i++) {
//         for (int j = 1; j <= prey_netsize; j++) {
//             Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
//             k++;
//         }
//     }
//     // Prey Sensor Weights
//     for (int i = 1; i <= prey_netsize*3; i++) {
//         Agent1.sensorweights[i] = phenotype(k);
//         k++;
//     }
//     // Fill World to Carrying Capacity
//     TVector<double> food_pos;
//     TVector<double> WorldFood(1, SpaceSize);
//     WorldFood.FillContents(0.0);
//     for (int i = 0; i <= start_CC; i++){
//         int f = rs.UniformRandomInteger(1,SpaceSize);
//         WorldFood[f] = 1.0;
//         food_pos.SetBounds(0, food_pos.Size());
//         food_pos[food_pos.Size()-1] = f;
//     }
//     // Make dummy predator list
//     TVector<double> pred_pos(0,-1);
//     TVector<Prey> preylist(0,0);
//     preylist[0] = Agent1;
//     // // Carrying capacity is 0 indexed, add 1 for true amount
//     // for (int i = 0; i < 200; i++){
//     //     double food_count = food_pos.Size();
//     //     double s_chance = 1 - food_count/(test_CC+1);
//     //     double c = rs.UniformRandom(0,1)*50;
//     //     if (c < s_chance){
//     //         int f = rs.UniformRandomInteger(1,SpaceSize);
//     //         WorldFood[f] = 1.0;
//     //         food_pos.SetBounds(0, food_pos.Size());
//     //         food_pos[food_pos.Size()-1] = f;
//     //     }
//     // }
//     for (int i = 0; i < start_prey_sim; i++){
//         Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_b_thresh);
//         newprey.Reset(rs.UniformRandomInteger(0,SpaceSize), 1.5);
//         newprey.NervousSystem = Agent1.NervousSystem;
//         newprey.sensorweights = Agent1.sensorweights;
//         preylist.SetBounds(0, preylist.Size());
//         preylist[preylist.Size()-1] = newprey;
//         }
//     for (double time = 0; time < PlotDuration*300; time += StepSize){
//         // Remove chomped food from food list
//         TVector<double> dead_food(0,-1);
//         for (int i = 0; i < food_pos.Size(); i++){
//             if (WorldFood[food_pos[i]] <= 0){
//                 dead_food.SetBounds(0, dead_food.Size());
//                 dead_food[dead_food.Size()-1] = food_pos[i];
//             }
//         }
//         if (dead_food.Size() > 0){
//             for (int i = 0; i < dead_food.Size(); i++){
//                 food_pos.RemoveFood(dead_food[i]);
//                 food_pos.SetBounds(0, food_pos.Size()-2);
//             }
//         }
//         // Carrying capacity is 0 indexed, add 1 for true amount
//         double c = rs.UniformRandom(0,1);
//         for (int i = 0; i < ((test_CC+1) - food_pos.Size()); i++){
//             double c = rs.UniformRandom(0,1);
//             if (c <= BT_G_Rate){
//                 int f = rs.UniformRandomInteger(1,SpaceSize);
//                 WorldFood[f] = 1.0;
//                 food_pos.SetBounds(0, food_pos.Size());
//                 food_pos[food_pos.Size()-1] = f;
//             }
//         }
//         // Prey Sense & Step
//         TVector<Prey> newpreylist;
//         TVector<int> preydeaths;
//         double total_state = 0;
//         for (int i = 0; i < preylist.Size(); i++){
//             preylist[i].Sense(food_pos, pred_pos);
//             preylist[i].Step(StepSize, WorldFood);
//             total_state += preylist[i].state;
//             if (preylist[i].birth == true){
//                 preylist[i].state = preylist[i].state - prey_repo;
//                 preylist[i].birth = false;
//                 Prey newprey(prey_netsize, prey_gain, prey_s_width, test_frate, test_feff, prey_BT_metaloss, prey_b_thresh);
//                 newprey.NervousSystem = preylist[i].NervousSystem;
//                 newprey.sensorweights = preylist[i].sensorweights;
//                 newprey.Reset(preylist[i].pos+2, prey_repo);
//                 newpreylist.SetBounds(0, newpreylist.Size());
//                 newpreylist[newpreylist.Size()-1] = newprey;
//             }
//             if (preylist[i].death == true){
//                 preydeaths.SetBounds(0, preydeaths.Size());
//                 preydeaths[preydeaths.Size()-1] = i;
//             }
//         }
//         // Update prey list with new prey list and deaths
//         if (preydeaths.Size() > 0){
//             for (int i = 0; i < preydeaths.Size(); i++){
//                 preylist.RemoveItem(preydeaths[i]);
//                 preylist.SetBounds(0, preylist.Size()-2);
//             }
//         }
//         if (newpreylist.Size() > 0){
//             for (int i = 0; i < newpreylist.Size(); i++){
//                 preylist.SetBounds(0, preylist.Size());
//                 preylist[preylist.Size()-1] = newpreylist[i];
//             }
//         }
//         ppfile << total_state << endl;
//         double total_food = 0;
//         for (int i = 0; i < WorldFood.Size();i++){
//             if (WorldFood[i] > 0){
//                 total_food += WorldFood[i];
//             }
//         }
//         fpfile << total_food << endl;
//         // Check Population Collapse
//         if (preylist.Size() <= 0){
//             break;
//         }
//         else{
//             newpreylist.~TVector();
//             preydeaths.~TVector();
//             dead_food.~TVector();
//         }
//     }
//     // Save
//     ppfile.close();
//     fpfile.close();
// }

// ================================================
// E. MAIN FUNCTION
// ================================================
int main (int argc, const char* argv[]) 
{
// ================================================
// EVOLUTION
// ================================================
	// long randomseed = static_cast<long>(time(NULL));
    // std::string fileIndex = "";  // Default empty string for file index
    // if (argc > 1) {
    //     randomseed += atoi(argv[1]);
    //     fileIndex = "_" + std::string(argv[1]); // Append the index to the file name
    // }

    // TSearch prey_s(PreyVectSize);
    // TSearch pred_s(PredVectSize);

    // std::ofstream preyevolfile;
    // std::string preyfilename = std::string("menagerie/LocalTest/prey_evolutions/evol" + fileIndex + ".dat"); // Create a unique file name
    // preyevolfile.open(preyfilename);
    // cout.rdbuf(preyevolfile.rdbuf());
    // std::ofstream predevolfile;
    // std::string predfilename = std::string("menagerie/LocalTest/pred_evolutions/evol" + fileIndex + ".dat"); // Create a unique file name
    // predevolfile.open(predfilename);
    // cout.rdbuf(predevolfile.rdbuf());
    // // Save the seed to a file
    // std::ofstream seedfile;
    // std::string seedfilename = std::string("menagerie/LocalTest/seeds/seed" + fileIndex + ".dat"); // Create a unique file name for seed
    // seedfile.open(seedfilename);
    // seedfile << randomseed << std::endl;
    // seedfile.close();
    // RandomState rs(randomseed);

    // // Configure the search
    // prey_s.SetRandomSeed(randomseed);
    // prey_s.SetSearchResultsDisplayFunction(ResultsDisplay);
    // prey_s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    // prey_s.SetSelectionMode(RANK_BASED);
    // prey_s.SetReproductionMode(CO_GENETIC_ALGORITHM);
    // prey_s.SetPopulationSize(PREY_POPSIZE);
    // prey_s.SetMaxGenerations(GENS);
    // prey_s.SetCrossoverProbability(CROSSPROB);
    // prey_s.SetCrossoverMode(UNIFORM);
    // prey_s.SetMutationVariance(MUTVAR);
    // prey_s.SetMaxExpectedOffspring(EXPECTED);
    // prey_s.SetElitistFraction(ELITISM);
    // prey_s.SetSearchConstraint(1);
    // prey_s.SetReEvaluationFlag(1);

    // pred_s.SetRandomSeed(randomseed);
    // pred_s.SetSearchResultsDisplayFunction(ResultsDisplay);
    // pred_s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
    // pred_s.SetSelectionMode(RANK_BASED);
    // pred_s.SetReproductionMode(CO_GENETIC_ALGORITHM);
    // pred_s.SetPopulationSize(PRED_POPSIZE);
    // pred_s.SetMaxGenerations(GENS);
    // pred_s.SetCrossoverProbability(CROSSPROB);
    // pred_s.SetCrossoverMode(UNIFORM);
    // pred_s.SetMutationVariance(MUTVAR);
    // pred_s.SetMaxExpectedOffspring(EXPECTED);
    // pred_s.SetElitistFraction(ELITISM);
    // pred_s.SetSearchConstraint(1);
    // pred_s.SetReEvaluationFlag(1);
    
    // for (int i = 0; i < swaps; i++){
    //     cout.rdbuf(preyevolfile.rdbuf());
    //     TVector<double> BOpred;
    //     BOpred.SetBounds(1, PredVectSize);
    //     BOpred = pred_s.BestIndividual();
    //     prey_s.SetSearchTerminationFunction(IntTerminationFunction);
    //     prey_s.SetCoEvaluationFunction(PreyTest);
    //     prey_s.SetBO(BOpred);
    //     TVector<double> bestotherprey = prey_s.BestOther();
    //     prey_s.ExecuteCoSearch(BOpred);
    //     TVector<double> bestpreyVector;
    //     ofstream BestPreyIndividualFile;
    //     TVector<double> preyphenotype;
    //     preyphenotype.SetBounds(1, PreyVectSize);
    //     // Save the genotype of the best individual
    //     // Use the global index in file names
    //     std::string bestpreyGenFilename = std::string("menagerie/LocalTest/prey_genomes/best.gen" + fileIndex + ".dat");
    //     std::string bestpreyNsFilename = std::string("menagerie/LocalTest/prey_nerves/best.ns" + fileIndex + ".dat");
    //     bestpreyVector = prey_s.BestIndividual();
    //     BestPreyIndividualFile.open(bestpreyGenFilename);
    //     BestPreyIndividualFile << bestpreyVector << endl;
    //     BestPreyIndividualFile.close();
    //     // Also show the best individual in the Circuit Model form
    //     BestPreyIndividualFile.open(bestpreyNsFilename);
    //     GenPhenMapping(bestpreyVector, preyphenotype, 0);
    //     BestPreyIndividualFile << preyphenotype << endl;
    //     BestPreyIndividualFile.close();

    //     // BO~TVector();
    //     cout.rdbuf(predevolfile.rdbuf());
    //     TVector<double> BOprey;
    //     BOprey.SetBounds(1, PreyVectSize);
        
    //     BOprey = prey_s.BestIndividual();

    //     pred_s.SetSearchTerminationFunction(IntTerminationFunction);
    //     pred_s.SetCoEvaluationFunction(PredTest);
    //     pred_s.SetBO(BOprey);
    //     TVector<double> bestotherpred = pred_s.BestOther();
    //     pred_s.ExecuteCoSearch(BOprey);
    //     TVector<double> bestpredVector;
    //     ofstream BestPredIndividualFile;
    //     TVector<double> predphenotype;
    //     predphenotype.SetBounds(1, PredVectSize);
    //     // Save the genotype of the best individual
    //     // Use the global index in file names
    //     std::string bestpredGenFilename = std::string("menagerie/LocalTest/pred_genomes/best.gen" + fileIndex + ".dat");
    //     std::string bestpredNsFilename = std::string("menagerie/LocalTest/pred_nerves/best.ns" + fileIndex + ".dat");
    //     bestpredVector = pred_s.BestIndividual();
    //     BestPredIndividualFile.open(bestpredGenFilename);
    //     BestPredIndividualFile << bestpredVector << endl;
    //     BestPredIndividualFile.close();
    //     // Also show the best individual in the Circuit Model form
    //     BestPredIndividualFile.open(bestpredNsFilename);
    //     GenPhenMapping(bestpredVector, predphenotype, 1);
    //     BestPredIndividualFile << predphenotype << endl;
    //     BestPredIndividualFile.close();

    //     // Destroy Vectors
    //     BOpred.~TVector();
    //     BOprey.~TVector();
    //     bestotherprey.~TVector();
    //     bestotherpred.~TVector();
    //     bestpreyVector.~TVector();
    //     bestpredVector.~TVector();
    //     preyphenotype.~TVector();
    //     predphenotype.~TVector();
    // }

    // return 0;

// ================================================
// RUN ANALYSES
// ================================================

    // // SET LIST OF AGENTS TO ANALYZE HERE, NEED LIST SIZE FOR INIT
    // TVector<int> analylist(0,1);
    // analylist.InitializeContents(15);

    // // // Behavioral Traces // // 
    // for (int i = 0; i < analylist.Size(); i++){
    //     int agent = analylist[i];
    //     // load the seed
    //     ifstream seedfile;
    //     double seed;
    //     seedfile.open("seed.dat");
    //     seedfile >> seed;
    //     seedfile.close();
    //     // load best prey
    //     ifstream prey_genefile;
    //     // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    //     prey_genefile.open("menagerie/TestBatch/prey_genomes/best.gen_%%.dat", agent);
    //     TVector<double> prey_genotype(1, PreyVectSize);
    //     prey_genefile >> prey_genotype;
    //     prey_genefile.close();
    //     // load best predator
    //     ifstream pred_genefile;
    //     // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    //     pred_genefile.open("menagerie/TestBatch/pred_genomes/best.gen_%%.dat", agent);
    //     TVector<double> pred_genotype(1, PredVectSize);
    //     pred_genefile >> pred_genotype;
    //     pred_genefile.close();
    //     // set the seed
    //     RandomState rs(seed);
    //     BehavioralTracesCoexist(prey_genotype, pred_genotype, rs, agent);
    // }

    // ANALYSES FOR JUST ONE AGENT
    // Set Batch and Agent for Analysis
    string batch = "TestBatch3";
    string agent = "4";
    // load the seed
    ifstream seedfile;
    double seed;
    seedfile.open("menagerie/" + batch + "/seeds/seed_" + agent + ".dat");
    seedfile >> seed;
    seedfile.close();
    // load best prey
    ifstream prey_genefile;
    // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    prey_genefile.open("menagerie/" + batch + "/prey_genomes/best.gen_" + agent + ".dat");
    TVector<double> prey_genotype(1, PreyVectSize);
    prey_genefile >> prey_genotype;
    prey_genefile.close();
    // load best predator
    ifstream pred_genefile;
    // SET WHICH BATCH YOU'RE EVALUATING HERE, CHECK ANALYSIS FUNCTIONS FOR THE SAME
    pred_genefile.open("menagerie/" + batch + "/pred_genomes/best.gen_" + agent + ".dat");
    TVector<double> pred_genotype(1, PredVectSize);
    pred_genefile >> pred_genotype;
    pred_genefile.close();
    // set the seed
    RandomState rs(seed);

    SSCoexist(prey_genotype, pred_genotype, rs, stoi(agent), batch);

    // // // Interaction Rate Collection // //
    // CollectEcoRates(genotype, rs);

    // // Code Testbed // // 
    // NewEco(genotype, rs);

    return 0;

}