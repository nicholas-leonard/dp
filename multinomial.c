#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define TRUE 1
#define FALSE 0
#define DEBUG 0
#define ERROR NULL
#define malloc2d(arr, h, w, t_type) ({ \
        arr = malloc((h) * sizeof(t_type *)); \
        for (int i=0; i<(h); i++) \
            {arr[i] = malloc((w) * sizeof(t_type *));} \
        })

/** for debugging */
void printDouble(int h, int w, double** array)
{    
    for (int i=0; i<h; i++)
    {
        for (int j=0; j<w; j++)
        {       
            printf("%f, ", array[i][j]);
        }
        printf("\n");
    }
}

/** for debugging */
void printInt(int h, int w, int** array)
{    
    for (int i=0; i<h; i++)
    {
        for (int j=0; j<w; j++)
        {
            printf("%d, ", array[i][j]);
        }
        printf("\n");
    }
}


/** destroy a 2d array */
void destroyArray(double** arr)
{
    free(*arr);
    free(arr);
}


/** update a row of cumulative distribution from prob distribution
@param:
    prob_width: width of prob_distr
    r: row of the cum_distr to be updated   
*/
void update_cum_row(double** cum_distr, double** prob_distr, int r, int prob_width)
{
        double accumulation = 0;
        for (int j=1; j<=prob_width; j++)
        {
            cum_distr[r][j] = accumulation + prob_distr[r][j-1];
            accumulation += prob_distr[r][j-1];
        }
        for (int j=1; j<=prob_width; j++)
        {
            if (accumulation > 0)
                cum_distr[r][j] /= accumulation;
        }
}


/** generate a random array from U~[0,1] of dim h*w */
double** rand_array(int h, int w)
{
    double** random_samples;
    malloc2d(random_samples, h, w, double);
    
    struct timeval utime;
    gettimeofday(&utime, NULL);
    
    srand(utime.tv_usec);
    for (int i=0; i<h; i++)
    {
        for (int j=0; j<w; j++)
        {
            random_samples[i][j] = rand() * 1.0 / RAND_MAX;
        }
    }
    return random_samples;
}


/** generate cumulative distribution from prob distribution
@param:
   h: height of prob_distr
   prob_width: width of prob_distr
*/
double** generate_cum_matrix(int h, int prob_width, double** prob_distr)
{
    double** cum_distr;
    malloc2d(cum_distr, h, prob_width+1, double);

    // Get normalized cumulative distribution from prob distribution
    for (int i=0; i<h; i++)
    {
        update_cum_row(cum_distr, prob_distr, i, prob_width);
    }
    return cum_distr;
}


/** Do a binary search for the slot in which the prob falls
@params:
    cum_width: width of the cum_distr
    row: the row of the cum_distr matrix to search where the prob falls
    uniform_sample: a sample from an uniform distribution U ~ [0, 1]
@return: the slot in which the prob falls, 
    ie cum_distr[row][slot] < prob < cum_distr[row][slot+1]
*/
int binarySearch(double** cum_distr, int cum_width, int row, double uniform_sample)
{
    int left_pointer = 0;
    int right_pointer = cum_width - 1;
    
    while(right_pointer - left_pointer > 1)
    {
        int left_tmp = left_pointer + (right_pointer - left_pointer) / 2;
        int right_tmp = right_pointer - (right_pointer - left_pointer) / 2;
        
        if (uniform_sample > cum_distr[row][left_tmp])
        {
            left_pointer = left_tmp;
        }
        
        if (uniform_sample < cum_distr[row][right_tmp])
        {
            right_pointer = right_tmp;
        }
    }
    return left_pointer;
}


/** Generate a subset of random samples from the prob_distr
@param:
    h: height of multinomial matrix, each row in matrix represents an individual experiment
    prob_width: width of prob_distr
    num_samples: number of samples to sample from the multinomial distribution
    with_replacement: 1 for true, 0 for false
@return:
    a sample of experts from the multinomial probability distribution
*/
int** multinomial(int h, int prob_width, double** prob_distr, int num_samples, int with_replacement)
{
    // return ERROR if the number of samples is larger than the size of multinomial distr
    if (num_samples > prob_width)
    {
        return ERROR;
    }
       
    double** cum_distr = generate_cum_matrix(h, prob_width, prob_distr);    

    // Generate 2d samples randomly from a uniform distribution ~ [0, 1]
    double** random_samples = rand_array(h, num_samples);
    
    if(DEBUG)
    {
        printf("prob_distr \n");
        printDouble(h, prob_width, prob_distr);
        printf("cum_distr \n");
        printDouble(h, prob_width+1, cum_distr);
        printf("random_samples \n");
        printDouble(h, num_samples, random_samples);
    }
    
    int** multinomial_samples;
    malloc2d(multinomial_samples, h, num_samples, int);
    
    // Allows drawn sample to be placed back into the pool for drawing again. ie with replacement
    if (with_replacement)
    {
        for (int i=0; i<h; i++)
        {
            for (int j=0; j<num_samples; j++)
            {
                // increase all the sample index by 1
                multinomial_samples[i][j] = 1 + binarySearch(cum_distr, prob_width + 1, i, random_samples[i][j]);
                if (DEBUG)
                {
                    printf("(%d, %d): random_sample %f in slot %d \n", i, j, \
                    random_samples[i][j], multinomial_samples[i][j]);
                }                                     
            }
        }
        destroyArray(cum_distr);
    }
    
    // A sample once is drawn will not be drawn again. ie without replacement
    else
    {
        for (int i=0; i<h; i++)
        {   
            for (int j=0; j<num_samples; j++)
            {
                if (DEBUG)
                {
                    printf("==before==\n");
                    printDouble(h, prob_width+1, cum_distr);
                } 
            
                int sample = binarySearch(cum_distr, prob_width + 1, i, random_samples[i][j]);
                // increase all the sample index by 1
                multinomial_samples[i][j] = sample + 1;
                prob_distr[i][sample] = 0;
                update_cum_row(cum_distr, prob_distr, i, prob_width);
                
                if (DEBUG)
                {
                    printf("==after==");
                    printf("(%d, %d): random_sample %f in slot %d \n", i, j, \
                    random_samples[i][j], multinomial_samples[i][j]);
                    printDouble(h, prob_width, prob_distr);
                    printDouble(h, prob_width+1, cum_distr);
                } 
            }
        }
        destroyArray(cum_distr);
    }
    
    destroyArray(random_samples);
    return multinomial_samples;
}

int main(void)
{
    int h = 5;
    int prob_width = 20;
    int num_samples = 20;
    
    double** prob_distr = malloc(h * sizeof(double*));
   
    for (int i=0; i<h; i++)
    {
        prob_distr[i] = malloc(prob_width * sizeof(double));
    }
    
    prob_distr[0][0] = 0.5;
    prob_distr[0][1] = 0.5;
    prob_distr[0][2] = 0.5;
    prob_distr[0][3] = 0.5;  
      
    prob_distr[1][0] = 0.5;
    prob_distr[1][1] = 0.5;
    prob_distr[1][2] = 0.5;
    prob_distr[1][3] = 0.5;
    
    prob_distr = rand_array(h, prob_width);
    
    int** mul_samples = multinomial(h, prob_width, prob_distr, num_samples, FALSE);
    printf("==final sample== \n");
    printInt(h, num_samples, mul_samples);
}




