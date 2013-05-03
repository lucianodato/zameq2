#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "lv2/lv2plug.in/ns/lv2core/lv2.h"

#define ZAMEQ2_URI "http://zamaudio.com/lv2/zameq2"

typedef enum {
	ZAMEQ2_INPUT  = 0,
	ZAMEQ2_OUTPUT = 1,

	ZAMEQ2_BOOSTDBL = 2,
	ZAMEQ2_SLOPEDBL = 3,
	ZAMEQ2_FREQL = 4,

	ZAMEQ2_BOOSTDB1 = 5,
	ZAMEQ2_Q1 = 6,
	ZAMEQ2_FREQ1 = 7,
	
	ZAMEQ2_BOOSTDB2 = 8,
	ZAMEQ2_Q2 = 9,
	ZAMEQ2_FREQ2 = 10,
	
	ZAMEQ2_BOOSTDBH = 11,
	ZAMEQ2_SLOPEDBH = 12,
	ZAMEQ2_FREQH = 13,

	ZAMEQ2_MLPTYPE = 14,
	ZAMEQ2_FREQMLP = 15	
	
} PortIndex;

typedef struct {
	float* input;
	float* output;

	float* boostdb1;
	float* q1;
	float* freq1;

	float* boostdb2;
	float* q2;
	float* freq2;

	float* boostdbl;
	float* slopedbl;
	float* freql;

	float* boostdbh;
	float* slopedbh;
	float* freqh;

	float*   mlptype;
	float* freqmlp;

	float srate;

	//lowshelf
	float zln1,zln2,zld1,zld2;
	float Bl[3];
	float Al[3];

	//peak 1
	float x1,x2,y1,y2;
	float a0x,a1x,a2x,b0x,b1x,b2x,gainx;
	
	//peak 2
	float x1a,x2a,y1a,y2a;
	float a0y,a1y,a2y,b0y,b1y,b2y,gainy;	

	//highshelf
	float zhn1,zhn2,zhd1,zhd2;
	float Bh[3];
	float Ah[3];

	//lowpass
	float a0lp,a1lp,a2lp,b1lp,b2lp;
	float x1lp,x2lp,y1lp,y2lp;

} ZamEQ2;



static LV2_Handle
instantiate(const LV2_Descriptor*     descriptor,
            double                    rate,
            const char*               bundle_path,
            const LV2_Feature* const* features)
{
	int i;
	ZamEQ2* zameq2 = (ZamEQ2*)malloc(sizeof(ZamEQ2));
	
	zameq2->srate = rate;
	
	//lp
	zameq2->a0lp=zameq2->a1lp=zameq2->a2lp=zameq2->b1lp=zameq2->b2lp=0.f;
	zameq2->x1lp=zameq2->x2lp=zameq2->y1lp=zameq2->y2lp=0.f;
	
	//peak 1
	zameq2->x1=zameq2->x2=zameq2->y1=zameq2->y2=0.f;
	zameq2->a0x=zameq2->a1x=zameq2->a2x=zameq2->b0x=zameq2->b1x=zameq2->b2x=zameq2->gainx=0.f;

	//peak 2
	zameq2->x1a=zameq2->x2a=zameq2->y1a=zameq2->y2a=0.f;
	zameq2->a0y=zameq2->a1y=zameq2->a2y=zameq2->b0y=zameq2->b1y=zameq2->b2y=zameq2->gainy=0.f;

	//Highshelf and Lowshelf
	zameq2->zln1=zameq2->zln2=zameq2->zld1=zameq2->zld2=0.f;
	zameq2->zhn1=zameq2->zhn2=zameq2->zhd1=zameq2->zhd2=0.f;
	
	for (i = 0; i < 3; ++i) {
		zameq2->Bl[i] = zameq2->Al[i] = zameq2->Bh[i] = zameq2->Ah[i] = 0.f;
		zameq2->Bl[i] = zameq2->Al[i] = zameq2->Bh[i] = zameq2->Ah[i] = 0.f;
		zameq2->Bl[i] = zameq2->Al[i] = zameq2->Bh[i] = zameq2->Ah[i] = 0.f;
	}

	return (LV2_Handle)zameq2;
}

static void
connect_port(LV2_Handle instance,
             uint32_t   port,
             void*      data)
{
	ZamEQ2* zameq2 = (ZamEQ2*)instance;

	switch ((PortIndex)port) {
	case ZAMEQ2_INPUT:
		zameq2->input = (float*)data;
		break;
	case ZAMEQ2_OUTPUT:
		zameq2->output = (float*)data;
		break;
	case ZAMEQ2_BOOSTDB1:
		zameq2->boostdb1 = (float*)data;
		break;
	case ZAMEQ2_Q1:
		zameq2->q1 = (float*)data;
		break;
	case ZAMEQ2_FREQ1:
		zameq2->freq1 = (float*)data;
		break;
	case ZAMEQ2_BOOSTDB2:
		zameq2->boostdb2 = (float*)data;
		break;
	case ZAMEQ2_Q2:
		zameq2->q2 = (float*)data;
		break;
	case ZAMEQ2_FREQ2:
		zameq2->freq2 = (float*)data;
		break;
	case ZAMEQ2_BOOSTDBL:
		zameq2->boostdbl = (float*)data;
		break;
	case ZAMEQ2_SLOPEDBL:
		zameq2->slopedbl = (float*)data;
		break;
	case ZAMEQ2_FREQL:
		zameq2->freql = (float*)data;
		break;
	case ZAMEQ2_BOOSTDBH:
		zameq2->boostdbh = (float*)data;
		break;
	case ZAMEQ2_SLOPEDBH:
		zameq2->slopedbh = (float*)data;
		break;
	case ZAMEQ2_FREQH:
		zameq2->freqh = (float*)data;
		break;
	case ZAMEQ2_MLPTYPE:
		zameq2->mlptype = (float*)data;
		break;
	case ZAMEQ2_FREQMLP:
		zameq2->freqmlp = (float*)data;
		break;
	}
}

// Works on little-endian machines only
static inline bool 
is_nan(float& value ) {
    if (((*(uint32_t *) &value) & 0x7fffffff) > 0x7f800000) {
      return true;
    }
    return false;
}

// Force already-denormal float value to zero
static inline void 
sanitize_denormal(float& value) {
    if (is_nan(value)) {
        value = 0.f;
    }
}

static inline int 
sign(float x) {
        return (x >= 0.f ? 1 : -1);
}

static inline float 
from_dB(float gdb) {
        return (exp(gdb/20.f*log(10.f)));
}

static inline float
to_dB(float g) {
        return (20.f*log10(g));
}

static void
activate(LV2_Handle instance)
{
}

//Orfanidis Peak filter (decramped)
static void
peq(float boostdb, float Q, float freq, float srate,float *a0, float *a1, float *a2, float *b0, float *b1, float *b2, float *gn) {

	float boost = from_dB(boostdb);
  	float fc = freq / srate;
	float w0 = fc*2.f*M_PI;
	float bwgain = (boostdb == 0.f) ? 1.f : (boostdb < 0.f) ? boost*from_dB(3.f) : boost*from_dB(-3.f);
	float bw = fc / Q;

	float G0=1.f; //dcgain
	float G=boost;
	float GB=bwgain;
	float Dw=bw;	

        float F,G00,F00,num,den,G1,G01,G11,F01,F11,W2,Dww,C,D,B,A;
        F = fabs(G*G - GB*GB);
        G00 = fabs(G*G - G0*G0);
        F00 = fabs(GB*GB - G0*G0);
        num = G0*G0 * (w0*w0 - M_PI*M_PI)*(w0*w0 - M_PI*M_PI)
                + G*G * F00 * M_PI*M_PI * Dw*Dw / F;
        den = (w0*w0 - M_PI*M_PI)*(w0*w0 - M_PI*M_PI)
                + F00 * M_PI*M_PI * Dw*Dw / F;
        G1 = sqrt(num/den);
        G01 = fabs(G*G - G0*G1);
        G11 = fabs(G*G - G1*G1);
        F01 = fabs(GB*GB - G0*G1);
        F11 = fabs(GB*GB - G1*G1);
        W2 = sqrt(G11 / G00) * tan(w0/2.f)*tan(w0/2.f);
        Dww = (1.f + sqrt(F00 / F11) * W2) * tan(Dw/2.f);
        C = F11 * Dww*Dww - 2.f * W2 * (F01 - sqrt(F00 * F11));
        D = 2.f * W2 * (G01 - sqrt(G00 * G11));
        A = sqrt((C + D) / F);
        B = sqrt((G*G * C + GB*GB * D) / F);
        *gn = G1;
        *b0 = (G1 + G0*W2 + B) / (1.f + W2 + A);
        *b1 = -2.f*(G1 - G0*W2) / (1.f + W2 + A);
        *b2 = (G1 - B + G0*W2) / (1.f + W2 + A);
        *a0 = 1.f;
        *a1 = -2.f*(1.f - W2) / (1.f + W2 + A);
        *a2 = (1 + W2 - A) / (1.f + W2 + A);

        sanitize_denormal(*b1);
        sanitize_denormal(*b2);
        sanitize_denormal(*a0);
        sanitize_denormal(*a1);
        sanitize_denormal(*a2);
        sanitize_denormal(*gn);
        if (is_nan(*b0)) { *b0 = 1.f; }
}

//Michale Massberg 1st and 2nd Order Lowpass Filter
static void
mlpeq(float freqcutoff,float fQ,float type,float srate,float *a0lp,float *a1lp,float *a2lp,float *b1lp,float *b2lp) {

	switch ((int)type){
	case 1:
		{
		float g1 = 2.0/pow((4.0 + pow(srate/freqcutoff,2)),0.5);

		float gm = fmaxf(pow((float)0.5, (float)0.5), pow(g1, (float)0.5));

		float wm = (2.0*M_PI*freqcutoff*pow(1 - gm*gm, (float)0.5))/gm;

		float Omega_m = tan(wm/(2.0*srate));

		float Omega_s = Omega_m*(pow((gm*gm - g1*g1)*((float)1.0 - gm*gm), (float)0.5))/(1.0 - gm*gm);
	
		float gamma = Omega_s + 1.0;
		float alpha0 = Omega_s + g1;
		float alpha1 = Omega_s - g1;
		float beta1 = Omega_s - 1.0;

		float a0 = alpha0/gamma;
		float a1 = alpha1/gamma;
		float b1 = beta1/gamma;

		*a0lp = a0;
		*a1lp = a1;
		*a2lp = 0.0f;
		*b1lp = b1;
		*b2lp = 0.0f;

		break;
		}	
	case 2:
		{

		// use same terms as in the picture
		float omegaC = 2.f*M_PI*(freqcutoff/srate);	
	
		float m = pow((pow(2.f, 0.5f)*M_PI)/ omegaC, 2.f);
		float n = pow(2.f*M_PI/ (fQ*omegaC), 2.f);
		float denom = pow(pow(2.f - m, 2.f) + n, 0.5f);

		// step 1: find g1 the gain at Nyquist
		float g1 = 2.f/denom;
	
		// branch on Q
		float omegaR = 0.f;
		float omegaS = 0.f;
		float omegaM = 0.f;

		// if > 0.707
		if(fQ > pow(0.5f, 0.5f)) 
		{
			// resonant gain (standard equation)
			float gr = 2.f*pow(fQ, 2.f)/ pow(4.f*pow(fQ, 2.f) - 1.f, 0.5f);
			float wr = omegaC*pow(1.f - (1.f/(2.f*fQ*fQ)) , 0.5f);
			omegaR = tan(wr/2.f); // NOTE this is wr/2fs in the paper - do fs cancel out from wn calc?

			float o = (gr*gr - g1*g1)/(gr*gr - 1.f);
			omegaS = omegaR*pow(o, 0.25f); // cube root
		}
		else
		{
			float a = 2.f - 1.f/ (fQ*fQ);
			float b = pow(1.f - (4.f*fQ*fQ)/ fQ*fQ*fQ*fQ + 4.f/g1, 0.5f);

			float coeff = pow((a+b)/2.f,0.5f);
			float wr = omegaC*coeff;

			omegaM = tan(wr/2.f);
			omegaS = (omegaC*pow((1.f - g1*g1), 0.25f))/2.f;
			omegaS = fminf(omegaS, omegaM);
		}

		// calc peak freq
		float wp = (2.f)*atan(omegaS);

		float q = pow(wp/omegaC, 2.f);
		float r = pow(wp/(fQ*omegaC), 2.f);

		// calc gain at POLE
		float gp = 1.f/pow( pow(1.f - q, 2.f) + r, 0.5f);
	
		// calculate gain at ZERO
		float wz = (2.f)*atan(omegaS/pow(g1, 0.5f));

		q = pow(wz/omegaC, 2.f);
		r = pow(wz/(fQ*omegaC), 2.f);

		// calc gain at ZERO
		float gz = 1.f/pow( pow(1.f - q, 2.f) + r, 0.5f);

		// calculalte required Q @ POLE
		float qp = pow((g1*(gp*gp - gz*gz))/ ((g1 + gz*gz)*(pow(g1 - 1.f, 2.f))), 0.5f);

		// calculalte required Q @ ZERO
		float num = g1*g1*(gp*gp - gz*gz);
		float den = gz*gz*(g1 + gp*gp)*(g1 - 1.f)*(g1 - 1.f);
		float qz = pow(num/den, 0.5f);

		// finally calc the coeffs
		float a0 = omegaS*omegaS + omegaS/qp + 1.f;
	
		float alpha0 = omegaS*omegaS + omegaS*pow(g1, 0.5f)/qz + g1;
		float alpha1 = 2.f*(omegaS*omegaS - g1);
		float alpha2 = omegaS*omegaS - omegaS*pow(g1, 0.5f)/qz + g1;

		float beta1 = 2.f*(omegaS*omegaS - 1.f);
		float beta2 = omegaS*omegaS - omegaS/qp + 1.f;

		*a0lp = alpha0/a0;
		*a1lp = alpha1/a0;
		*a2lp = alpha2/a0;
		*b1lp = beta1/a0;
		*b2lp = beta2/a0;

		break;
		}
	}

	sanitize_denormal(*b1lp);
	sanitize_denormal(*b2lp);
	sanitize_denormal(*a0lp);
	sanitize_denormal(*a1lp);
	sanitize_denormal(*a2lp);

}

//Butterworth Shelf Filters
static void
bw_shelfeq(float boostdb, float freq,float q,float type, float srate,float B[], float A[]) {

	float w0 = 2.f*M_PI*freq/ srate;
	float G = boostdb;

	float alpha,b0,b1,b2,a0,a1,a2;
	G = powf(10.f,G/20.f); 
	float AA  = sqrt(G);
	
	if (type == 0) {
		//lowshelf
		alpha = sin(w0)/2.f * sqrt( (AA + 1.f/AA)*(1.f/q - 1.f) + 2.f );
		b0 =    AA*( (AA+1.f) - (AA-1.f)*cos(w0) + 2.f*sqrt(AA)*alpha );
		b1 =  2.f*AA*( (AA-1.f) - (AA+1.f)*cos(w0)                   );
		b2 =    AA*( (AA+1.f) - (AA-1.f)*cos(w0) - 2.f*sqrt(AA)*alpha );
		a0 =        (AA+1.f) + (AA-1.f)*cos(w0) + 2.f*sqrt(AA)*alpha;
		a1 =   -2.f*( (AA-1.f) + (AA+1.f)*cos(w0)                   );
		a2 =        (AA+1.f) + (AA-1.f)*cos(w0) - 2.f*sqrt(AA)*alpha;
	
	}
	else
	{
		//highshelf
		alpha = sin(w0)/2.f * sqrt( (AA + 1.f/AA)*(1.f/q - 1.f) + 2.f );
		b0 =    AA*( (AA+1.f) + (AA-1.f)*cos(w0) + 2.f*sqrt(AA)*alpha );
		b1 =  -2.f*AA*( (AA-1.f) + (AA+1.f)*cos(w0)                   );
		b2 =    AA*( (AA+1.f) + (AA-1.f)*cos(w0) - 2.f*sqrt(AA)*alpha );
		a0 =        (AA+1.f) - (AA-1.f)*cos(w0) + 2.f*sqrt(AA)*alpha;
		a1 =   2.f*( (AA-1.f) - (AA+1.f)*cos(w0)                   );
		a2 =        (AA+1.f) - (AA-1.f)*cos(w0) - 2.f*sqrt(AA)*alpha;

	}

	B[0] = b0/a0;
	B[1] = b1/a0;
	B[2] = b2/a0;
	A[0] = 1.f;
	A[1] = a1/a0;
	A[2] = a2/a0;
}

//Direct Form I - y(n) = a0x(n) + a1x(n-1) + a2x(n-2) - b1y(n-1) - b2y(n-2)
static void
calculate_directformI(float *input,float *x1,float *x2,float *y1,float *y2,float b0,float b1,float b2,float a1,float a2){

	float tmp = 0.f;	
	
	tmp = *input * b0 + 
	*x1 * b1 +
	*x2 * b2 -
	*y1 * a1 -
	*y2 * a2;
	*x2 = *x1;
	*y2 = *y1;
	*x1 = *input;
	*y1 = tmp;
	*input = tmp;
}

static void
run(LV2_Handle instance, uint32_t n_samples)
{
	ZamEQ2* zameq2 = (ZamEQ2*)instance;

	const float* const input  = zameq2->input;
	float* const       output = zameq2->output;

	const float        boostdb1 = *(zameq2->boostdb1);
	const float        q1 = *(zameq2->q1);
	const float        freq1 = *(zameq2->freq1);
	
	const float        boostdb2 = *(zameq2->boostdb2);
	const float        q2 = *(zameq2->q2);
	const float        freq2 = *(zameq2->freq2);
	
	const float        boostdbl = *(zameq2->boostdbl);
	const float        slopedbl = *(zameq2->slopedbl);
	const float        freql = *(zameq2->freql);

	const float        boostdbh = *(zameq2->boostdbh);
	const float        slopedbh = *(zameq2->slopedbh);
	const float        freqh = *(zameq2->freqh);

	const int          mlptype = *(zameq2->mlptype);
	const float        freqmlp = *(zameq2->freqmlp);

	//this is a flag that indicates that if some filter is active so it doesn't enter the for cycle in case is not
	bool is_active_flag = false;

	//Lowshelf
	if (boostdbl !=0){//Avoids calcs if gain is 0
	bw_shelfeq(boostdbl,freql,slopedbl,0,zameq2->srate,zameq2->Bl,zameq2->Al);
	is_active_flag = true;//some filter is active
	}
	//Peak 1
	if (boostdb1 !=0){//Avoids calcs if gain is 0	
	peq(boostdb1,q1,freq1,zameq2->srate,&zameq2->a0x,&zameq2->a1x,&zameq2->a2x,&zameq2->b0x,&zameq2->b1x,&zameq2->b2x,&zameq2->gainx);
	is_active_flag = true;//some filter is active
	}
	//Peak 2
	if (boostdb2 !=0){//Avoids calcs if gain is 0
	peq(boostdb2,q2,freq2,zameq2->srate,&zameq2->a0y,&zameq2->a1y,&zameq2->a2y,&zameq2->b0y,&zameq2->b1y,&zameq2->b2y,&zameq2->gainy);
	is_active_flag = true;//some filter is active
	}
	//Highshelf
	if (boostdbh !=0){//Avoids calcs if gain is 0
	bw_shelfeq(boostdbh,freqh,slopedbh,1,zameq2->srate,zameq2->Bh,zameq2->Ah);
	is_active_flag = true;//some filter is active
	}
	//Low Pass with no resonance
	if (mlptype !=0){//if filter is not bypassed
	mlpeq(freqmlp,0.71f,mlptype,zameq2->srate,&zameq2->a0lp,&zameq2->a1lp,&zameq2->a2lp,&zameq2->b1lp,&zameq2->b2lp);
	is_active_flag = true;//some filter is active
	}

	if (is_active_flag == true){
		for (uint32_t pos = 0; pos < n_samples; pos++) {
	
			float in = input[pos];

			sanitize_denormal(zameq2->x1);
			sanitize_denormal(zameq2->x2);
			sanitize_denormal(zameq2->y1);
			sanitize_denormal(zameq2->y2);
			sanitize_denormal(zameq2->x1a);
			sanitize_denormal(zameq2->x2a);
			sanitize_denormal(zameq2->y1a);
			sanitize_denormal(zameq2->y2a);
			sanitize_denormal(zameq2->x1lp);
			sanitize_denormal(zameq2->x2lp);
			sanitize_denormal(zameq2->y1lp);
			sanitize_denormal(zameq2->y2lp);
			sanitize_denormal(zameq2->zln1);
			sanitize_denormal(zameq2->zln2);
			sanitize_denormal(zameq2->zld1);
			sanitize_denormal(zameq2->zld2);
			sanitize_denormal(zameq2->zhn1);
			sanitize_denormal(zameq2->zhn2);
			sanitize_denormal(zameq2->zhd1);
			sanitize_denormal(zameq2->zhd2);
			sanitize_denormal(in);

			//Cascade filters Using a function for Direct Form I this way filters could be bypassed
		
			//lowshelf
			if (boostdbl !=0){//Avoids processing if gain is 0
			calculate_directformI(&in,&zameq2->zln1,&zameq2->zln2,&zameq2->zld1,&zameq2->zld2,zameq2->Bl[0],zameq2->Bl[1],zameq2->Bl[2],zameq2->Al[1],zameq2->Al[2]);
			}
			//parametric1
			if (boostdb1 !=0){//Avoids processing if gain is 0
			calculate_directformI(&in,&zameq2->x1,&zameq2->x2,&zameq2->y1,&zameq2->y2,zameq2->b0x,zameq2->b1x,zameq2->b2x,zameq2->a1x,zameq2->a2x);
			}
			//parametric2
			if (boostdb2 !=0){//Avoids processing if gain is 0
			calculate_directformI(&in,&zameq2->x1a,&zameq2->x2a,&zameq2->y1a,&zameq2->y2a,zameq2->b0y,zameq2->b1y,zameq2->b2y,zameq2->a1y,zameq2->a2y);
			}
			//highshelf
			if (boostdbh !=0){//Avoids processing if gain is 0
			calculate_directformI(&in,&zameq2->zhn1,&zameq2->zhn2,&zameq2->zhd1,&zameq2->zhd2,zameq2->Bh[0],zameq2->Bh[1],zameq2->Bh[2],zameq2->Ah[1],zameq2->Ah[2]);
			}
			//lowpass
			if (mlptype !=0){//if filter is not bypassed
			calculate_directformI(&in,&zameq2->x1lp,&zameq2->x2lp,&zameq2->y1lp,&zameq2->y2lp,zameq2->a0lp,zameq2->a1lp,zameq2->a2lp,zameq2->b1lp,zameq2->b2lp);
			}
			//finally
			output[pos]=in;

		}
	}
}

static void
deactivate(LV2_Handle instance)
{
}

static void
cleanup(LV2_Handle instance)
{
	free(instance);
}

const void*
extension_data(const char* uri)
{
	return NULL;
}

static const LV2_Descriptor descriptor = {
	ZAMEQ2_URI,
	instantiate,
	connect_port,
	activate,
	run,
	deactivate,
	cleanup,
	extension_data
};

LV2_SYMBOL_EXPORT
const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
	switch (index) {
	case 0:
		return &descriptor;
	default:
		return NULL;
	}
}
