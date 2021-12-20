
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <boost/numeric/ublas/matrix.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <thread>
#include <random>

using namespace std;
enum {C=500};

vector<string> Menu();
void Read_instance(string adress, int D[C][C]);
void Write_instance(vector<int> M, int Of, string name);

class PROBLEM {
    public:
        vector<int> M_best;
        int Of_best;

        PROBLEM() {
            Of_best = 0;
            generator.seed(static_cast<unsigned> (std::chrono::steady_clock::now().time_since_epoch().count()));
            begin = chrono::steady_clock::now();
        }

        void GRASP(int D[C][C], int m, int time_max, string name){
            //This resolution of time is needed because the program takes a millisecond-order time to find a solution.

            while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - begin).count() < time_max) {

                Construction_Greedy(D, m);
                Local_Search(D, m);

                //CUIDADO alpha fijo.
                if (Of > Of_best) {
                    M_best = M;
                    Of_best = Of;
                }
                Of = 0;
                M.clear();
                cont = Cont();
            }
            Write_instance(M_best, Of_best, name);
        }

        void TABU(int D[C][C], int m, int time_max, string name) {
            Construction_Random(D, m);
            Tabu_Search(D,m,time_max);
            Write_instance(M_best, Of_best, name);
        }



    private:
        mt19937 generator;
        vector<int> RCL, M;
        int flag = 0, val=0, Of=0;
        chrono::steady_clock::time_point begin;

        class Cont {

            /*
            This class is used for managing the Contribution and CL, within the Of all at the same.
            The decision was to make the Cont list variable, because as a con we have to look for the
            index we want to remove each time, but managing an array of fixed length, would imply a
            search for a maximum and minimum with more elements than the strictly necessary. We stop looking for a number
            when we find it, but we don't do that when calculating a max or min.
            */

            public:
                pair<vector<int>, vector<int>> CONT;

                void init() {

                    //Initialization of the Cont array with ordered vertices and 0 contributions.

                    for (int i = 0; i < C; i++) {
                        CONT.first.push_back(i);
                        CONT.second.push_back(0);
                    }
                }

                void init(vector<int> M, int D[C][C]) {

                    //Initialization of the cont vector with an alreagy generated M, it excludes the M values from the CL
                    // and computes the contributions.

                    for (int i = 0; i < C; i++) {
                        if ((find(M.begin(), M.end(), i) == M.end()) && M.back()!=i) {
                            CONT.first.push_back(i);
                            CONT.second.push_back(0);

                            for (int j = 0; j < M.size(); j++) {
                                CONT.second.back()+=D[M[j]][i];
                            }
                        }
                    }
                }

                void update(int index, int D[C][C], int& Of, vector<int>& M) {

                    /*
                    This function takes care of the CL, Of and the Contrib at the same time. The CONT is thought as
                    a pair of arrays, one with the indices in CL and other with the contribution values. Notice that 
                    at the first iteration none contribution is added at Of, with is passed by value to the function.
                    Also is worth minding that we are managing here two vectors, and because of that the iterators are
                    incompatible and we must operate with indexes.
                    */

                    M.push_back(index);

                    int j = distance(CONT.first.begin(),find(CONT.first.begin(), CONT.first.end(), M.back()));
                    Of += CONT.second[j];
                    CONT.first.erase(CONT.first.begin()+j);
                    CONT.second.erase(CONT.second.begin()+j);
                
                    for (int i = 0; i < CONT.second.size(); i++) {
                        CONT.second[i] += D[M.back()][CONT.first[i]];
                    }

                }

                void update(int index_in, int index_out, int D[C][C], int& Of, vector<int>& M, int val) {
                    /*
                    This function takes two indexes, one from CL and another form M and swap them , updating the contribution value 
                    and the objective function.
                    */

                    int flag = M[index_out];
                    M[index_out] = CONT.first[index_in];
                    CONT.first[index_in] = flag;
                    Of += CONT.second[index_in]-D[CONT.first[index_in]][M[index_out]]-val;
                    CONT.second[index_in] = val;
                
                    for (int i = 0; i < CONT.second.size(); i++) {
                        CONT.second[i] += D[M[index_out]][CONT.first[i]] - D[CONT.first[index_in]][CONT.first[i]];
                    }
                }

                /*
                The min and max functions return the min and max values respectively, it is important to mention
                that the pointer * method is used to obtain the value of the max or min instead of the iterator.
                */

                int max() {
                    return *max_element(CONT.second.begin(), CONT.second.end());
                }

                int min() {
                    return *min_element(CONT.second.begin(), CONT.second.end());
                }
        };

        Cont cont;

        void Construction_Greedy(int D[C][C], int m) {

            //The construction of the solution by pseudo greedy method controlled by alpha constant, by default 0.6141.

            cont.init();
            int max = 0;
            pair<int, int> index;
            const double alpha = 0.6141;

            for (int i = 0; i < C; i++) {
                flag = distance(D[i], max_element(D[i], D[i] + C)); 
                if (max < D[i][flag]) {
                    index.first = flag;
                    index.second = i;
                    max = D[i][flag];
                }
            }

            cont.update(index.first, D, Of, M);
            cont.update(index.second, D, Of, M);

            while (M.size() < m) {
                RCL.clear();
                for (int i = 0; i < cont.CONT.first.size(); i++) {
                    if (cont.CONT.second[i] >= cont.max() - alpha * (cont.max() - cont.min())) {
                        RCL.push_back(cont.CONT.first[i]);
                    }
                }
                //Not uniform distribution
                uniform_int_distribution<int> distribution(0, RCL.size()-1);
                cont.update(RCL[distribution(generator)], D, Of, M);
                //CL integrada en la Cont
            }
        }

        void Construction_Random(int D[C][C], int m) {

            /*
            Random construction of M, feasible solution, by shuffling with an uniform distribution
            and selecting the m first elements. After this, the initialization overload of update is used
            to update cont.
            */


            for (int i = 0; i < C; i++) {
                M.push_back(i);
            }

            shuffle(M.begin(), M.end(), generator);
            M.erase(M.begin() + m, M.end());
            cont.init(M, D);

            for (int i = 0; i < M.size(); i++) {
                for (int j = 0; j < M.size(); j++) {
                    Of += D[M[i]][M[j]];
                }
            }
            Of = Of / 2;
        }

        void Local_Search(int D[C][C], int m) {

            //The local search of GRASP.

            int index;
            flag = 0;

            for (int i = 0; i < M.size(); i++) {

                val = 0;
                for (int j = 0; j < M.size(); j++) {
                    val += D[M[i]][M[j]];
                }

                flag = 0;
                index = 0;

                while ((flag == 0) && (index<cont.CONT.second.size())) {
                    
                    if ((cont.CONT.second[index]-D[cont.CONT.first[index]][M[i]]) > val) {
                        cont.update(index, i, D, Of, M, val);
                        flag = 1;
                        i = 0;
                    }
                    index++;
                }
            }
        }

        void Tabu_Search(int D[C][C], int m, int time_max) {

            //Tabu search with default tabu tenure of 5. The tabu list is an array allocated in the Heap.

            const int tabu_tenure = 5;
            int iteration = tabu_tenure+1, max, val;
            auto* TABU = new int [C] {0};
           
            auto valid = [&TABU, &iteration, tabu_tenure](int i) {
                return ((iteration - TABU[i]) > tabu_tenure);
            };

            while (chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - begin).count() < time_max) {

                for (int i = 0; i < M.size(); i++) {
                    max = distance(cont.CONT.first.begin(), find_if(cont.CONT.first.begin(), cont.CONT.first.end(),valid));
                    val = D[M[i]][cont.CONT.first[max]];

                    for (int j = 0; j < cont.CONT.first.size(); j++) {
                        if ((valid(cont.CONT.first[j])) && ((cont.CONT.second[max]-val)<(cont.CONT.second[j]-D[M[i]][cont.CONT.first[j]]))) {
                            max = j;
                            val = D[M[i]][cont.CONT.first[max]];
                        }
                    }

                    val = 0;
                    for (int j = 0; j < M.size(); j++) {
                        val += D[M[i]][M[j]];
                    }

                    TABU[M[i]]=iteration;
                    cont.update(max, i, D, Of, M, val);
                    iteration++;
                    if (Of_best < Of) {
                        M_best = M;
                        Of_best = Of;
                    }

                }   
            }
            delete[] TABU;
        }

};


int main() {
    string adress;
    vector<string> answer;
    int m, index=0, timer,opt;
    chrono::steady_clock::time_point init = chrono::steady_clock::now();
    
    //The matrix of distances is allocated into the Heap, because of its dimensions.

    auto* D = new int[C][C];

    answer = Menu();
    m = stoi(answer.rbegin()[2]);
    timer = stoi(answer.rbegin()[1]);
    opt = stoi(answer.rbegin()[0]);
    PROBLEM instance;

    if (answer.size() == 4) {
        adress = answer.front();
        Read_instance(adress, D);
        
        if (opt == 0) {
            instance.GRASP(D, m, timer, "result");
        }
        else if (opt == 1) {
            instance.TABU(D, m, timer, "result");
        }

    }
    else{

        for (int j = 0; j < answer.size()-3; j++) {
            Read_instance(answer[j], D);
            if (opt == 0) {
                instance.GRASP(D, m, timer, "grasp_" + to_string(timer) + "s_" + to_string(j));
            }
            else if (opt == 1) {
                instance.TABU(D, m, timer, "tabu_" + to_string(timer) + "s_" + to_string(j));
            }
        }


    }
    
    delete[] D;

    return 0;
}

vector<string> Menu() {
    /* 
    Basic Menu for selecting the files of the directory to apply the algorithm 

    The instances must be in csv format and of CxC size. This C is a global variable defined above, 
    aplying the algorithm to another problem size consists in simply changing the C-value. For C++ 
    construction reasons this value can't be given by the User and has to be changed manually.
    */

    string adress, choice, choice2, m, timer, opt;
    vector<string> files, output;
    bool run = false;

    do {
        cout << "Please introduce the path of the directory containing the instances" << endl <<
            "The instances must be in csv format and the directory must have the form C:\\my_path\\files, with a size of "
            + to_string(C) + " variables" << endl << endl;

        getline(cin, adress);

    } while (!filesystem::exists(adress));

    for (const auto& entry : filesystem::directory_iterator(adress)) {
        files.push_back(entry.path().generic_string());
    }
    cout<< endl << "These are the available files, please select one typing the number of the option" << endl << endl;

    for (int i = 0; i < files.size(); i++) {
        cout << "Option " + to_string(i) + " - " << files[i] << endl;
    }
    cout << "Option " + to_string(files.size()) + "-Run all the files in the folder, this may take aa while"<<endl;
    do {

        cin >> choice;

        do {
            cout << endl << "Please type the number of seconds you want to run the code" << endl << endl;
            cin >> timer;
        } while (stoi(timer)<=0);

        do {
            cout << endl << "Please type the algorithm to use" << endl << "Option 0 - GRASP"<<endl<< "Option 1 - Tabu"<<endl<<endl;
            cin >> opt;
        } while (stoi(opt) != 0 && stoi(opt)!=1);

        do {
            cout << endl << "Please type the size of the subset wanted for the solution. Mind that m<"+to_string(C) << endl << endl;
            cin >> m;
        } while (stoi(m) >= C);

        if (stoi(choice) <= files.size()-1) {

            cout<< endl << "The solver is about to run with " + files[stoi(choice)] << endl << "Are you sure?, type y for yes and n for not" << endl<< endl;
            cin >> choice2;
            if (choice2 == "y") {
                run = true;
                output.push_back(files[stoi(choice)]);
            }
            else {
                cout << "Please select another option" <<endl<< endl;
            }
            
        }
        else if (stoi(choice) == files.size()) {
            cout << endl << "The solver is about to run with all the files in the directory" << endl << "Are you sure?, type y for yes and n for not" << endl << endl;
            cin >> choice2;
            if (choice2 == "y") {
                run = true;
                output = files;
            }
            else {
                cout << "Please select another option" << endl << endl;
            }  
        }
        else {
            cout<< endl << "You didn't enter a correct number, please mind it has to be between 0 and" + to_string(files.size()-1) << endl;
        }
    } while (run == false);

    output.push_back(m);
    output.push_back(timer);
    output.push_back(opt);
    return output;

    
}

void Read_instance(string adress, int D[C][C]) {
    /* 
    This function simply reads the MDS distance from the file passed by the Menu() function, that means, the one 
    chosen by the User. For this reading, the csv format with ';' is needed, because the delimeter used to 
    look for variables is ';'.
    
    Also the function is concieved to read (MS-DOS)csv files exported from Excel, where each row is separated form the others
    by a linebreak, is because of this that the use of two loops is needed to read the file, one for each row and then 
    other to read columns of one specific row.

    If the given matrix is not of the kind CxC, the program execution will end.
    */

    vector<int> d;
    string input, input2;

    ifstream fin;
    fin.open(adress);

    int row = 0, column = 0;
    while (getline(fin,input)) {
        stringstream ss(input);
        while (getline(ss, input2, ';')) {
            D[row][column]=stoi(input2);
            column++;
        }
        column = 0;
        row++;
    }

    fin.close();

    if (0) {
        cout << "El tamaño de la matriz de distancias es [" + to_string(d.size()) + "x" + to_string(d.size()) +
            " .Cuando debería ser de " + to_string(C) + "x" + to_string(C) << endl;
    }

}

void Write_instance(vector<int> M, int Of, string name) {

    //This function performs the task of saving results into the project folder. 
    //It is a long process and loosing results is expensive.

    ofstream fout;
    fout.open(name + ".csv", ios::app);
    for (int i = 0; i < M.size(); i++) {
        fout << M[i];
        fout << ';';
    }
    fout << Of<<endl;
    fout.close();
}

