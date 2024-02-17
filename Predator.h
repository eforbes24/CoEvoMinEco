#pragma once

#include "CTRNN.h"
#include "Prey.h"

// The Predator class declaration

class Predator {
	public:
		// The constructor
		
		Predator(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh, double handling_time,double pred_PCOT_scalar, double pred_NCOT_scalar, double pred_NCOT_thresh)
		{
			Set(networksize, gain, s_width, frate, feff, metaloss, birth_thresh, handling_time, pred_PCOT_scalar, pred_NCOT_scalar, pred_NCOT_thresh);
		};
		Predator() = default;
		// The destructor
		~Predator() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};

		// Control
        void Set(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double b_thresh, double handling_time, double pred_PCOT_scalar, double pred_NCOT_scalar, double pred_NCOT_thresh);
		void Reset(double initpos, double initstate);
		void Sense(TVector<double> &prey_loc, TVector<double> &food_pos, bool hutchflag, int hutchL, int hutchR);
		void Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist, bool hutchflag, int hutchL, int hutchR);

		int size;
		double pos, gain, f_sensor, p_sensor, h_sensor, s_width, pastpos,frate, handling_time, mov_dist, PCOT, NCOT, NCOT_thresh,
		handling_counter, munchrate, birthrate, snackflag, birth_thresh, feff, metaloss, state, fitness;
		bool handling, birth, death;
		TVector<double> sensorweights;
		CTRNN NervousSystem;
		Prey prey;
};