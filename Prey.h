#pragma once

#include "CTRNN.h"

// The Prey class declaration

class Prey {
	public:
		// The constructor
		Prey(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh, double prey_PCOT_scalar, double prey_NCOT_scalar, double prey_NCOT_thresh)
		{
			Set(networksize, gain, s_width, frate, feff, metaloss, birth_thresh, prey_PCOT_scalar, prey_NCOT_scalar, prey_NCOT_thresh);
		};
		Prey() = default;
		// The destructor
		~Prey() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetSensorWeight(int to, double value) {sensorweights[to] = value;};
		void SetSensorState(double fstate, double pstate) {f_sensor = fstate;
			p_sensor = pstate;};

		// Control
        void Set(int networksize, double gain, double s_width, double frate, double feff, double metaloss, double birth_thresh, double prey_PCOT_scalar, double prey_NCOT_scalar, double prey_NCOT_thresh);
		void Reset(double initpos, double initstate);
		void Sense(TVector<double> &food_pos, TVector<double> &pred_loc, bool hutchflag, int hutchL, int hutchR);
		void Step(double StepSize, TVector<double> &WorldFood);

		int size;
		double pos, gain, f_sensor, p_sensor, h_sensor, s_width, pastpos, state, frate, feff, metaloss, birth_thresh,
		munchrate, birthrate, snackflag, mov_dist, PCOT, NCOT, NCOT_thresh, fitness;
		bool death, birth;
		TVector<double> sensorweights;
		CTRNN NervousSystem;
};
