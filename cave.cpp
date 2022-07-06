#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>
#include <allegro5/allegro5.h>
#include <allegro5/allegro_primitives.h>

#define Rows 240
#define Cols 360
#define Steps 3500
#define Square 3
#define Procs 6
#define waterPercentage 70
#define stepDelay 50
#define graphic

//Per comodità nei vari test tutto ciò che riguarda l'aspetto grafico verrà dichiarato solo
//se 'graphic' è definito

//Definizione della struttura di ogni cella:
// -Ogni cella ha esattamente 8 bit di dimensione
// -I primi 2 bit si utilizzeranno per tenere traccia dell'acqua
// -I successivi 3 per tenere traccia del sedimento
// Nonostante i bit necessari siano solo 5 putroppo in memoria non si possono salvare dati
// più piccoli di un byte

typedef union cell{
    struct { uint8_t water:2; uint8_t sediment:3; } inside;

    unsigned char all;
}cell;

cell* readMatrix = new cell[Rows*(Cols/Procs+2)];
cell* writeMatrix  = new cell[Rows*(Cols/Procs+2)];
int rank, left, right;
int rank_border_left, rank_border_right;

inline void swap(){ cell* tmp; tmp = readMatrix; readMatrix = writeMatrix; writeMatrix = tmp;}

inline void init();
#ifdef graphic
    void print(int step, cell* matrix);
    ALLEGRO_DISPLAY* display;
    ALLEGRO_EVENT_QUEUE* queue;
    ALLEGRO_EVENT event;
#endif

inline void sendBorders();

inline void receiveBorders();

inline void transFunction();

inline bool amIRock(int i, int j);

inline char amIWater(int i, int j);

inline void iAmAir(int i, int j);

inline void transFunctionBorders();

inline void dropWater();

#define Read(i,j) readMatrix[coordsToIndex(i,j)].inside
#define Write(i,j) writeMatrix[coordsToIndex(i,j)].inside

// Funzione per convertire le coordinate matriciali in un indice poichè  per motivi di efficienza
// le strutture utilizzate dei vettori
inline int coordsToIndex(int i, int j) { return ((j*Rows)+i);}

MPI_Datatype columnType;
MPI_Datatype subMatrixType;
MPI_Comm cave;

int main(int argc, char* argv[])
{
    int size;
    int source, dest;
    double starttime;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD , &rank);
    MPI_Comm_size( MPI_COMM_WORLD , &size);
    if(rank==0)
        starttime=MPI_Wtime();

    //Definisco i datatype, uno dei quali utilizzo esclusivamente per inviare le sottomatrici ad un unico processo
    //poichè allegro permette solamente la stampa su display da un solo processo
    MPI_Type_contiguous(Rows , MPI_BYTE , &columnType);
    #ifdef graphic
    MPI_Type_contiguous(Rows*(Cols/Procs), MPI_BYTE, &subMatrixType);
    MPI_Type_commit( &subMatrixType);
    #endif
    MPI_Type_commit(&columnType);

    //definisco la topologia che verrà utilizzata per mappare i processi
    int dims[1] = {size};
    int periods[1] = {0};
    MPI_Cart_create( MPI_COMM_WORLD , 1 , dims , periods , 1 , &cave);
    MPI_Cart_shift( cave , 0 , 1 , &left , &right);
    const int left_coords[] = {0};
    const int right_coords[] = {Procs-1};
    MPI_Cart_rank( cave , left_coords , &rank_border_left);
    MPI_Cart_rank( cave , right_coords , &rank_border_right);

    #ifdef graphic
    cell* completeMatrix;
    if(rank==0)
    {
        al_init();
        display = al_create_display(Cols*Square, Rows*Square);
        al_init_primitives_addon();
        al_set_app_name("Cave");
        queue = al_create_event_queue();
        al_register_event_source(queue, al_get_display_event_source(display));
        cell* tmp = new cell[Rows*Cols];
        completeMatrix = tmp;
    }
    #endif

    //Inizializzazione della matrice di lettura
    for(int i=0; i<Rows; i++)
        for(int j=1; j< Cols/Procs+1; ++j)
            readMatrix[coordsToIndex(i,j)].all=0;

    //Definizione di una semplice configurazione iniziale
    init();
    srand(time(NULL) + rank);
    int stop = 0, step = 0;
    
    //for(int step = 0; step < Steps; ++step)
    while( stop != 1 && step < Steps)
    {
        //la funzione dropWater ha una possibilità predefinita di generare acqua
        dropWater();
        #ifdef graphic
        if(step%stepDelay==0)
        {   
            //Ogni processo invia la sua sottomatrice al processo 0 che si occuperà della stampa
            //N.B. 0 non è necessariamente il master Thread, poichè la topologia ha riordinato i rank per ottimizzare
            MPI_Gather(&readMatrix[coordsToIndex(0,1)] , 1 , subMatrixType , 
                        completeMatrix , 1 , subMatrixType , 0 , cave);
            if(rank==0)
                print(step, completeMatrix);
        }
        #endif

        //Invio ai vicini i miei bordi
        sendBorders();

        //Nel frattempo elaboro la funzione di transizione per le celle interne
        transFunction();

        //Ricevo i bordi dei vicini nelle halo cells
        receiveBorders();

        //Essendo le Recv bloccanti una volta arrivato qui so che posso elaborare anche i bordi
        transFunctionBorders();

        //Faccio lo swap tra matrice di scrittura e lettura
        swap();

        #ifdef graphic
        if(rank == 0)
        {
            al_peek_next_event(queue, &event);
            if(event.type == ALLEGRO_EVENT_DISPLAY_CLOSE)
                stop = 1;
        }
        MPI_Bcast( &stop , 1 , MPI_INT , 0 , cave);
        #endif
        step++;

        #ifndef graphic
        if(step >= Steps)
            break;
        #endif
    }
    MPI_Barrier( cave);
    if(rank==0)
        printf("Time: %2.5f\n", MPI_Wtime()-starttime);

    #ifdef graphic
    while(stop != 1)
    {
        if(rank == 0)
        {
            al_peek_next_event(queue, &event);
            if(event.type == ALLEGRO_EVENT_DISPLAY_CLOSE)
                stop = 1;
        }
        MPI_Bcast( &stop , 1 , MPI_INT , 0 , cave);
    }
    #endif
    MPI_Type_free(&columnType);

    #ifdef graphic
    if(rank == 0)
        delete[] completeMatrix;
    MPI_Type_free(&subMatrixType);
    #endif

    delete[] readMatrix;
    delete[] writeMatrix;

    #ifdef graphic
    if(rank == 0)
    {
        al_destroy_event_queue(queue);
        al_destroy_display(display);
        al_uninstall_system();
    }
    #endif
    MPI_Finalize();

    return 0;
}


inline void sendBorders()
{
    MPI_Request request;
    if(rank==rank_border_left)
    {
        MPI_Isend( &readMatrix[coordsToIndex(0, Cols/Procs)], 1, columnType, right, 1, cave, &request);
    }
    else
    {
        if(rank==rank_border_right)
            MPI_Isend( &readMatrix[coordsToIndex(0,1)] , 1 , columnType , left , 0 , cave , &request);
        else
        {
            MPI_Isend( &readMatrix[coordsToIndex(0, Cols/Procs)], 1, columnType, right, 1, cave, &request);
            MPI_Isend( &readMatrix[coordsToIndex(0,1)] , 1 , columnType , left , 0 , cave, &request);
        }

    }
}

inline void receiveBorders()
{
    MPI_Status status;
    if(rank==rank_border_left)
    {
        MPI_Recv( &readMatrix[coordsToIndex(0, Cols/Procs +1)] , 1 , columnType , right , 0 , cave , &status);
    }else{
        if(rank==rank_border_right)
            MPI_Recv( &readMatrix[coordsToIndex(0,0)] , 1 , columnType , left , 1 , cave , &status);
        else
        {
            MPI_Recv( &readMatrix[coordsToIndex(0, Cols/Procs +1)], 1, columnType, right, 0, cave, &status);
            MPI_Recv( &readMatrix[coordsToIndex(0,0)] , 1 , columnType , left , 1 , cave, &status);
        }
    }
}

inline void init()
{
    for(int j=1; j<Cols/Procs+1; ++j)
    {
        Read(0,j).sediment=6;
        Read(Rows-1,j).sediment=6;
        Read(1,j).sediment=6;
        Read(Rows-2,j).sediment=6;
    }
}

inline void transFunction()
{
    for(int i=0; i<Rows; ++i)
    {
        for(int j=2; j<Cols/Procs; ++j)
        {   
            //Se la cella è una roccia viene automaticamente settata tale evitando tutti i controlli
            if(amIRock(i,j))
                continue;
            
            //controllo se nella cella c'è acqua
            switch (amIWater(i, j))
            {
            case 'C':
                continue;
                break;
            
            case 'F':
                break;

            }
            //se arrivo a questo punto significa che nella cella c'è aria
            iAmAir(i, j);
        }
    }
}

inline void transFunctionBorders()
{
    for(int i=0; i<Rows; ++i)
    {
        for(int j=1; j<Cols/Procs+1; j+=Cols/Procs-1)
        {
            //Se la cella è una roccia viene automaticamente settata tale evitando tutti i controlli
            if(amIRock(i,j))
                continue;
            
            //controllo se nella cella c'è acqua
            switch (amIWater(i, j))
            {
            case 'C':
                continue;
                break;
            
            case 'F':
                break;

            }
            //se arrivo a questo punto significa che nella cella c'è aria
            iAmAir(i, j);
        }
    }
}

inline bool amIRock(int i, int j)
{
    if(Read(i,j).sediment>5)
    {
        Write(i,j).sediment = 6;
        Write(i,j).water = 0;
        return true;
    }
    return false;
}

inline char amIWater(int i, int j)
{
    if(Read(i,j).water > 0)
    {
        int r = 0, a = 0, rocks = 0;
        char caso;
                
        for(int x = -1; x<2; ++x)
        {
            for(int y = -1; y<2; ++y)
            {   
                if((x!=0 || y!=0) && Read(i+y,j+x).sediment > 5)
                    r++;
                else if((x!=0 || y!=0) && Read(i+y,j+x).water== 0) a++;
            }
            if(x!=0 && Read(i,j+x).sediment > 5 || Read(i+x,j).sediment > 5)
                rocks++;
            bool f=false;
            if(x!=0 && Read(i-1,j+x).sediment>5 && Read(i,j+x).sediment>5 && Read(i+1,j).sediment>5)
                f=true;
            if(f)
                Write(i,j).sediment++;
        }

        if(rocks>0)
        {
            Write(i,j).sediment += 1;
            if(Read(i+1,j).sediment>5 && (Read(i+1,j+1).sediment<5 && Read(i+1,j+1).water == 0 && 
               Read(i+1,j-1).sediment<5 && Read(i+1,j-1).water == 0))
                Write(i,j).sediment += 2;
            if((Read(i-1,j).water==0 && Read(i+1,j).water==0) && (Read(i,j-1).sediment>5 || 
                Read(i,j+1).sediment>5))
                Write(i,j).sediment++;    
        }
        if(r==8)
        {
            Write(i,j).water=1;
            Write(i,j).sediment++;
            return 'C';
        }
        if(Read(i+1,j).sediment>5 && Read(i+1,j+1).sediment>5 && Read(i+1,j-1).sediment>5)
        {
            Write(i,j).water=1;
            return 'C';
        }

        if(Read(i-1,j).water > 0) caso = 'W';
        else 
        {
            if(Read(i-1,j).sediment<6 || Read(i+1,j).water > 0) caso = 'A';
            if(Read(i-1,j).sediment>5) caso = 'R';
        }
        switch(caso)
        {
            case 'W':
                Write(i,j).water += 1;
                break;
                    
            case 'A':
                if(Write(i,j).water > 0)
                        Write(i,j).water -= 1;
                break;

            case 'R':
            {
                if(a+r == 8 && Read(i,j-1).sediment<6 && Read(i,j+1).sediment<6)
                {
                    Write(i,j).water=1;
                    break;
                }
                rocks = 0;
                bool air = false;
                for(int x = -1; x<2; ++x)
                {   
                    if(x!=0 && Read(i-1,j+x).water > 0)
                        Write(i,j).water += 1;
                    if(Read(i+1,j+x).water > 0)
                        air = true;
                            
                    for(int y = -1; y<2; ++y)
                    {
                        if((x!=0 || y!=0) && Read(i+y,j+x).sediment > 5)
                            rocks++;
                                
                    }
                }
                if(rocks>1 && air)
                    if(Write(i,j).water > 0)
                        Write(i,j).water -= 1;
                break; 
            }        
        }
        return 'C';
    }
    return 'F';
}

inline void iAmAir(int i, int j)
{
    if(Read(i-1,j).sediment>5)
    {
        int count=0;
        for(int x=-1 ; x<2; x+=2)
            if(Read(i-1,j+x).water>0)
                count++;
        if(count>0)
            Write(i,j).water+=count;
        else
            Write(i,j).water = 0;
    }
    else 
    {   
        if(Read(i+1,j).sediment>5 && Read(i-1,j).water == 0)
        {
            int count=0;
            for(int x=-1 ; x<2; x+=2)
                if(Read(i-1,j+x).water>0 && Read(i,j+x).sediment>5)
                    count++;
            if(count>0)
                Write(i,j).water+=count;
        }
        else
        {
            int count=0;
            for(int x=-1 ; x<2; x+=2)
                if(Read(i-1,j+x).water>0 && Read(i,j+x).sediment>5)
                    if(rand()%5 == 1) count++;
            if(Read(i-1,j).water>1 || (Read(i-1,j).water>0 && (Read(i,j-1).sediment>5 ||
               Read(i,j+1).sediment >5)) || count>0) Write(i,j).water+=1;
            else Write(i,j).water = 0;
        }
    }
}

inline void dropWater()
{
    //genero casualmente gocce d'acqua, lo spawn è regolato da una percentuale
    int a =  rand()%waterPercentage;
    if(a >waterPercentage-30)
    {
        //scelgo la colonna in cui generare l'acqua, e scendo di riga finche non trovo una cella dove non c'è roccia
        int xW = (rand()%(Cols/Procs))+1;
        for(int y = 2; y<Rows; ++y)
            if(Read(y,xW).sediment<6)
            {
                Read(y,xW).water+=1;
                return;
            }
    }
}

#ifdef graphic
void print(int step, cell* matrix)
{
    //stampa con allegro la matrice facendo riferimento a 3 stati differenti di una cella: aria, acqua, roccia
    printf("\n---%d---", step);
    al_clear_to_color(al_map_rgb(0, 0, 0));
    for(int i = 0; i<Rows; ++i)
    {
        for(int j=0; j<Cols; ++j)
        {
            if(matrix[coordsToIndex(i,j)].inside.sediment > 5)
                al_draw_filled_rectangle(j*Square, i*Square, j*Square+Square, i*Square+Square, al_map_rgb(89, 44, 16));
            else
            {   if(matrix[coordsToIndex(i,j)].inside.water > 0)
                    al_draw_filled_rectangle(j*Square, i*Square, j*Square+Square, i*Square+Square, al_map_rgb(37, 150, 207));
                else
                    al_draw_filled_rectangle(j*Square, i*Square, j*Square+Square, i*Square+Square, al_map_rgb(173, 173, 173));
            }
        }
    }
    al_flip_display();
    al_rest(0.01);
}
#endif


//mpic++ cave.cpp -o a.out -I/usr/include/allegro5 -L/usr/lib -lallegro -lallegro_primitives