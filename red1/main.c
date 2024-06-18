//
//  main.c
//  red1
//
//  Created by toby on 12.06.24.
//


#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#include "ocl.h"


//width (check kernel!)
const int w = 32;


//testing reduction - in place
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //ocl
    struct ocl_obj ocl;
    ocl_init(&ocl);
    
    //vars
    int n = 123456789;

    /*
     ===========
     init
     ===========
     */
    
    //buf
    cl_mem uu = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, n*sizeof(float), NULL, &ocl.err);
    
    //args
    ocl.err = clSetKernelArg(ocl.vec_ini, 0, sizeof(cl_mem), (void*)&uu);
    
    //init
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vec_ini, 1, NULL, (size_t*)&n, NULL, 0, NULL, NULL);
    
    /*
     ===========
     loop1
     ===========
     */
    
    int nele = n;

    for(int i=0; i<10; i++)
    {
        int nsub = ceil((float)nele/(float)w);       //number subtotals
        int s = pow(w,i);                            //stride
        
        //prc dims
        size_t prc_n = w*nsub;                      //padded
        size_t prc_w = w;
        
        
        printf("loop %d %10d %10d %10d %10zu %10d\n", i, n, nele, nsub, prc_n, s);
        
        //args
        ocl.err = clSetKernelArg(ocl.vec_sum, 0, sizeof(size_t), (void*)&n);
        ocl.err = clSetKernelArg(ocl.vec_sum, 1, sizeof(size_t), (void*)&s);
        ocl.err = clSetKernelArg(ocl.vec_sum, 2, sizeof(cl_mem), (void*)&uu);
    
        //calc - pad
        ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vec_sum, 1, NULL, &prc_n, &prc_w, 0, NULL, NULL);
        
        //exit
        if(nsub==1) break;
        
        //iter
        nele = nsub;
        
    }
    
    
    //result
    float r;
    
    //read
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, uu, CL_TRUE, 0, sizeof(float), &r, 0, NULL, NULL);

    //disp
    printf("ana %d\n", n);
//    printf("ana %f\n", 0.5*n*(n-1));
    printf("num %f\n", r);
    

    
    /*
     ===========
     clean
     ===========
     */
    
    //free
    ocl.err = clReleaseMemObject(uu);

    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}





/*
 
 
 //testing reduction - subtotals
 int main(int argc, const char * argv[])
 {
     printf("hello\n");
     
     //ocl
     struct ocl_obj ocl;
     ocl_init(&ocl);
     
     //vars
     int n = 123456789; //(float loses precision above 8 digits)
     
     //width (check kernel!)
     int w = 256;
     

      //===========
      //init
      //===========

     
     //buf
     cl_mem uu = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, n*sizeof(float), NULL, &ocl.err);
     
     //args
     ocl.err = clSetKernelArg(ocl.vec_ini, 0, sizeof(cl_mem), (void*)&uu);
     
     //init
     ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vec_ini, 1, NULL, &n, NULL, 0, NULL, NULL);
     
     
      //===========
      //loop1
      //===========
      
     
     int n1 = n;
     cl_mem u1 = uu;
     cl_mem u2;
     

     for(uint i=0; i<10; i++)
     {
         int n2 = ceil((float)n1/(float)w);     //output size
         int n3 = n2*w;                           //proc size (pad)
         
         printf("loop %12zu %12zu %12zu\n", n1, n2, n2*w);
         
         //output
         u2 = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, n2*sizeof(float), NULL, &ocl.err);
     
         //args
         ocl.err = clSetKernelArg(ocl.vec_sub, 0, sizeof(int),    (void*)&n1);
         ocl.err = clSetKernelArg(ocl.vec_sub, 1, sizeof(cl_mem), (void*)&u1);
         ocl.err = clSetKernelArg(ocl.vec_sub, 2, sizeof(cl_mem), (void*)&u2);
         
         //calc - pad processes
         ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vec_sub, 1, NULL, &n3, &w, 0, NULL, NULL);
         
         //read
         if(n2==1)
         {
             //result
             float r;
             
             //read
             ocl.err = clEnqueueReadBuffer(ocl.command_queue, u2, CL_TRUE, 0, sizeof(float), &r, 0, NULL, NULL);
         
             //disp
             printf("%f\n", r);
             
             break;
         }
         
         //increment
         n1 = n2;
         u1 = u2;
         
         //free (not the first one)
         if(i>0)
         {
             ocl.err = clReleaseMemObject(u2);
         }
     }
     
     
      //===========
      //clean
      //===========
     
     //free
     ocl.err = clReleaseMemObject(uu);

     //clean
     ocl_final(&ocl);
     
     printf("done\n");
     
     return 0;
 }

 
 
 */
