#include<stdio.h>
#include <cmath>
#include<math.h>

#define DEC_DIGITS 4
class MP_float{
    private:
    long long int num;
    long long int denom;
    
    public:
    /* constructors. */
    __host__ __device__ MP_float(){
    
    }
    
    __host__ __device__ MP_float(float f){
        this->denom = 1;
        for(int i=0; i<DEC_DIGITS; i++){
            this->denom *= 10;
            f *= 10;
        }
        float rem = f-(int)f;
        this->num = (int)f;
        if(rem>0.5){
           this->num++;
        }
    }
    
    __host__ __device__ MP_float(const MP_float &i_f){
        this->num = i_f.num;
        this->denom = i_f.denom;
    }
    
    __host__ __device__ MP_float operator-(){
        return -(this->num);
    }
    
    __host__ __device__ MP_float operator+(MP_float intf){
        MP_float sum;
        int this_denom_greater = (this->denom>intf.denom);
        while(this->denom != intf.denom){
              if(this_denom_greater){
                  intf.denom *= 10;
                  intf.num *= 10;
              }
              else{
                  this->denom *= 10;
                  this->num *= 10;
              }
        }

        sum.num = this->num + intf.num;
        sum.denom = this->denom;
        return sum;
    }
    
    __host__ __device__ MP_float operator-(MP_float intf){
        MP_float dif;
        int this_denom_greater = (this->denom>intf.denom);
        while(this->denom != intf.denom){
              if(this_denom_greater){
                  intf.denom *= 10;
                  intf.num *= 10;
              }
              else{
                  this->denom *= 10;
                  this->num *= 10;
              }
        }
        dif.num = this->num - intf.num;
        dif.denom = this->denom;
        return dif;
    }
    
    __host__ __device__ MP_float operator*(MP_float intf){
        MP_float prod;
        MP_float temp1 = *this;
        MP_float temp2 = intf;
        if(log10f(this->denom)>=8){
            
            for(int i=0; i<4; i++){
                temp1.num /= 10;
                temp1.denom/= 10;
            }
        }
        if(log10f(intf.denom)>=8){
            
            for(int i=0; i<4; i++){
                temp2.num /= 10;
                temp2.denom/= 10;
            }
        }
        prod.num = temp1.num * temp2.num;
        prod.denom = temp1.denom * temp2.denom;
        return prod;
    }
    
    __host__ __device__ bool operator<=(MP_float intf){
        MP_float a = *this;
        MP_float b = intf;
        while(a.denom > b.denom){
            a.num /= 10;
            a.denom /= 10;
        }
        while(a.denom < b.denom){
            b.num /= 10;
            b.denom /= 10;
        }
        if(a.num > b.num){
             return false;
        }
        if(a.num == b.num){
            if(this->denom < intf.denom){
                return false;
            }
            return true;
        }
        return true;
        
    }
    
    
    __host__ __device__ int get_num(){
        return this->num;
    }
    
    __host__ __device__ int get_denom(){
        return this->denom;
    }
    
    __host__ __device__ float get_float(){
        float result = 0;
        result = (double)this->num/(double)this->denom;
        return result;
    }
    
    __host__ __device__ void print(){
        printf("%d/%d\n",this->num,this->denom);
    }
};
