//
//  prg.cl
//  red1
//
//  Created by toby on 12.06.24.
//

//width
constant int w = 4;


//init
kernel void vec_ini(global float *uu)
{
    int glb_pos = get_global_id(0);
    
    uu[glb_pos] = glb_pos;

    return;
}


//sum in place
kernel void vec_sum(const  int      n,      //actual size
                    const  int      s,      //stride
                    global float    *uu)
{
    int glb_pos = get_global_id(0);
    int loc_pos = get_local_id(0);
    int grp_pos = get_group_id(0);
    
//    int glb_dim = get_global_size(0);
//    int loc_dim = get_local_size(0);
//    int grp_dim = get_num_groups(0);

//    printf("%d/%d %d/%d %d/%d\n", glb_pos, glb_dim, loc_pos, loc_dim, grp_pos, grp_dim);


    //buffer
    local float uu_loc[w];
    
    //read (zero padded values)
    uu_loc[loc_pos] = (glb_pos<n)*uu[glb_pos];
    
    //sync
    mem_fence(CLK_LOCAL_MEM_FENCE);
    
    float usum = 0e0f;
    
    //reduce
    for(int i=1; i<w; i++)
    {
        usum += uu_loc[i];
    }
    
    //write all
//    uu[glb_pos] = (loc_pos==0)?usum:0e0f;
    
    //write stride
    if(loc_pos==0)
    {
        uu[glb_pos] = usum;
    }

    printf("%3d %3d %2d %2d %2d %8.4f %8.4f %8.4f\n", n, s, glb_pos, loc_pos, grp_pos, uu_loc[loc_pos], usum, uu[glb_pos]);
    
    
    return;
}







//sum to vector of subtotals
kernel void vec_sub(const  int      n,
                    global float   *uu,
                    global float   *vv)
{
    //width
    const int w = 4;
    
    int glb_pos = get_global_id(0);
    int loc_pos = get_local_id(0);
    int grp_pos = get_group_id(0);
    
//    int glb_dim = get_global_size(0);
//    int loc_dim = get_local_size(0);
//    int grp_dim = get_num_groups(0);

//    printf("%d/%d %d/%d %d/%d\n", glb_pos, glb_dim, loc_pos, loc_dim, grp_pos, grp_dim);

    //buffer
    local float ww[w];
    
    //zero padding
    ww[loc_pos] = (glb_pos<n)*uu[glb_pos];
    
    //sync
    mem_fence(CLK_LOCAL_MEM_FENCE);

    //reduce
    for(uint i=1; i<w; i*=2)
    {
        if((loc_pos%(2*i))==0)
        {
            ww[loc_pos] += ww[loc_pos+i];
//            ww[loc_pos+i] = 0e0f;
        }
        //sync
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //write
    if(loc_pos==0)
    {
        vv[grp_pos] = ww[0];
        
//        printf("%3d %2d  %2d %e\n", glb_pos, loc_pos, grp_pos, vv[grp_pos]);
    }
    
    return;
}






