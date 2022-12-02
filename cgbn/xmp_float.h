#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <gmp.h>
#include <cmath>
#include "cgbn.h"
#include <inttypes.h>


#define TPI 4
#define BITS 512

typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

class MP_float {
   private:
      cgbn_mem_t<BITS> num;
      //cgbn_mem_t<BITS> denom;
      unsigned int denom; 
      //char padding[20];

   public:   
      __host__ __device__ MP_float(float a) {
          for(int i=0; i<(BITS+31)/32; i++){
              this->num._limbs[i] = 0;
              //this->denom._limbs[i] = 0;
          }
          int sign = 0;
          uint32_t value = 0;
          uint32_t exp = 1;
          if(a < 0){
              sign = 1;
          }
          float temp = a;

          int temp_int = temp;
          float mod = temp - temp_int;
          while(fabs(mod) > 0){
              temp *= 10;
              temp_int = temp;
              mod = temp - temp_int;
              exp += 1;
          }
          if(sign){
              value = -temp;
          }
          else{
              value = temp;
          }
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 1);                                 // create a CGBN context
          env_t bn_env(bn_context); 
          env_t::cgbn_t numerator, num_float;
          env_t::cgbn_t num_float_signed;
          cgbn_set_ui32(bn_env, num_float, 1);
          cgbn_set_ui32(bn_env, numerator, value);
          //cgbn_set_ui32(bn_env, denom, exp);
          this->denom = exp;
          //printf("*********num:%d,denom:%d\n",cgbn_get_ui32(bn_env, numerator),cgbn_get_ui32(bn_env,denom));
          //bn_env.div(num_float, numerator, denom);
          if(sign){
              bn_env.negate(num_float_signed, numerator);
              cgbn_store(bn_env, &(this->num), num_float_signed);
              

          }
          else{
              cgbn_store(bn_env, &(this->num), numerator);
          }
          //cgbn_store(bn_env, &(this->denom), denom);
      }
      
      __host__ __device__ MP_float(const MP_float &a){
          this->num = a.num;
          this->denom = a.denom;
          //this->report = a.report;
      }
      
      __host__ __device__ MP_float operator+ (MP_float num1) {
          MP_float result(0.0);
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 0);                                 // create a CGBN context
          env_t bn_env(bn_context);                       // construct a bn environment for 1024 bit math
          env_t::cgbn_t a, b, r, const_10;
          cgbn_set_ui32(bn_env, const_10, 10);
          bn_env.load(a, &(this->num));
          //bn_env.load(a_denom, &(this->denom));

          bn_env.load(b, &(num1.num));
          //bn_env.load(b_denom, &(num1.denom));
          
          int a_denom_greater = (this->denom>num1.denom);
          while(this->denom != num1.denom){
              //printf("hi");
              if(a_denom_greater){
                  //bn_env.mul(b_denom, b_denom, const_10);
                  num1.denom += 1;
                  bn_env.mul(b, b, const_10);
              }
              else{
                  //bn_env.mul(a_denom, a_denom, const_10);
                  this->denom += 1;
                  bn_env.mul(a, a, const_10);
              }
          }
          
          bn_env.add(r, a, b);
          bn_env.store((&result.num), r);
          result.denom = num1.denom;
          //bn_env.store((&result.denom), b_denom);

          return result;
      }
      
      __host__ __device__ MP_float operator- (MP_float num1) {
          MP_float result(0.0);
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 0);                                 // create a CGBN context
          env_t bn_env(bn_context);                       // construct a bn environment for 1024 bit math
          env_t::cgbn_t a, b, r, const_10;
          cgbn_set_ui32(bn_env, const_10, 10);
          bn_env.load(a, &(this->num));
          //bn_env.load(a_denom, &(this->denom));

          bn_env.load(b, &(num1.num));
          //bn_env.load(b_denom, &(num1.denom));
          
          int a_denom_greater = (this->denom>num1.denom);
          while(this->denom != num1.denom){
              //printf("hi");
              if(a_denom_greater == 1){
                  //bn_env.mul(b_denom, b_denom, const_10);
                  num1.denom += 1;
                  bn_env.mul(b, b, const_10);
              }
              else{
                  //bn_env.mul(a_denom, a_denom, const_10);
                  this->denom += 1;
                  bn_env.mul(a, a, const_10);
              }
          }
          
          bn_env.sub(r, a, b);
          bn_env.store((&result.num), r);
          result.denom = num1.denom;
         // bn_env.store((&result.denom), b_denom);

          return result;
      }
      
      __host__ __device__ MP_float operator* (MP_float num1) {
          MP_float result(0.0);
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 0);                                 // create a CGBN context
          env_t bn_env(bn_context);                       // construct a bn environment for 1024 bit math
          env_t::cgbn_t a, b, r, const_10;

          cgbn_set_ui32(bn_env, const_10, 10);
          bn_env.load(a, &(this->num));
          //bn_env.load(a_denom, &(this->denom));

          bn_env.load(b, &(num1.num));
          //bn_env.load(b_denom, &(num1.denom));
          
          bn_env.mul(r, a, b);
          result.denom = this->denom + num1.denom;
          bn_env.store((&result.num), r);
          //bn_env.store((&result.denom), r_denom);

          return result;
      }
      
       __host__ __device__ bool operator<= (MP_float num1) {
          MP_float result(0.0);
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 0);                                 // create a CGBN context
          env_t bn_env(bn_context);                       // construct a bn environment for 1024 bit math
          env_t::cgbn_t a, b, const_10;
          unsigned int denom_diff = fabs(this->denom-num1.denom);
          cgbn_set_ui32(bn_env, const_10, 10);
          bn_env.load(a, &(this->num));
          //bn_env.load(a_denom, &(this->denom));

          bn_env.load(b, &(num1.num));
          //bn_env.load(b_denom, &(num1.denom));
          if(this->denom>num1.denom){
              for(int i=0; i<denom_diff; i++){
                  bn_env.div(a, a, const_10);
              }
          }
          else{
              for(int i=0; i<denom_diff; i++){
                  bn_env.div(b, b, const_10);
              }
          }
          
         
          
          bool res = bn_env.compare(b, a);
          if(res>0){
              return true;
          }   
           return false;
          }
         
      uint32_t* get_limbs(){
          return this->num._limbs;
      }
      
      __host__ __device__ double get_float(){
          cgbn_error_report_t *report;
          context_t bn_context(cgbn_report_monitor, report, 0);                                 // create a CGBN context
          env_t bn_env(bn_context);                       // construct a bn environment for 1024 bit math
          env_t::cgbn_t double_num, double_num_unsigned, const_10, const_0;
          cgbn_set_ui32(bn_env, const_10, 10);
          cgbn_set_ui32(bn_env, const_10, 0);
          
          int sign = 0;
          bn_env.load(double_num, &(this->num));
          bn_env.load(double_num_unsigned, &(this->num));
          //bn_env.load(double_denom, &(this->denom));
          if(bn_env.compare(double_num, const_0)>0){
              sign = 1;
              bn_env.negate(double_num_unsigned,double_num);
              printf("negating");
          }
          bn_env.store(&(this->num), double_num_unsigned);
          //bn_env.store(&(this->denom), double_denom);
          while((this->num._limbs[1]!=0 && (int)this->num._limbs[1]!=-1)){
              bn_env.load(double_num, &(this->num));
              //bn_env.load(double_denom, &(this->denom));

              bn_env.div(double_num, double_num, const_10);
              //bn_env.div(double_denom, double_denom, const_10);
              this->denom --;
              bn_env.store(&(this->num), double_num);
              //bn_env.store(&(this->denom), double_denom);
              //printf("get double whileee: %d, %d, ",num._limbs[1], denom._limbs[2]);
          }
          uint32_t num = 0;
          //num += this->num._limbs[1];
          //num = num>>32;
          num += this->num._limbs[0];
          uint32_t denom = 0;
          //denom += this->denom._limbs[1];
          //denom = denom>>32;
          denom += this->denom;
          double result = (double)num/(double)denom;
          if(sign == 1){
              result = -result;
          }
          printf("get double: %d, %d, %f",num, denom, result);
          return result;

      }
      
};