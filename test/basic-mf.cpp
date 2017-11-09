/** 

���־���R���� 

   D1 D2 D3 D4 

U1 5  3  -  1 

U2 4  -  -  1 

U3 1  1  -  5 

U4 1  -  -  4 

U5 -  1  5  4 

***/ 

#include<iostream> 

#include<cstdio> 

#include<cstdlib> 

#include<cmath> 

using namespace std; 

 

void matrix_factorization(double *R,double *P,double *Q,int N,int M,int K,int steps=5000,float alpha=0.0002,float beta=0.02) 

{ 

 for(int step =0;step<steps;++step) 

 { 

  for(int i=0;i<N;++i) 

  { 

   for(int j=0;j<M;++j) 

   { 

    if(R[i*M+j]>0) 

    { 

     //�������error ���ǹ�ʽ6�����e(i,j) 

     double error = R[i*M+j]; 

     for(int k=0;k<K;++k) 

      error -= P[i*K+k]*Q[k*M+j]; 

 

     //���¹�ʽ6 

     for(int k=0;k<K;++k) 

     { 

      P[i*K+k] += alpha * (2 * error * Q[k*M+j] - beta * P[i*K+k]); 

      Q[k*M+j] += alpha * (2 * error * P[i*K+k] - beta * Q[k*M+j]); 

     } 

     } 

    } 

   } 
   if (step == 0){

   cout << "---" <<endl;

		 for(int i=0;i<N;++i) 
		{
		  	for(int j=0;j<K;++j) {
		//   		P[i*K+j]=rand()%9/float(10); 
//				P[i*K+j]=1;
		   		cout <<  P[i*K+j]<<',';
			}
			cout<<endl;
		}
	   cout << "---" <<endl;

		 for(int i=0;i<M;++i) 
		{
		  	for(int j=0;j<K;++j) {
		//   		P[i*K+j]=rand()%9/float(10); 
//				P[i*K+j]=1;
		   		cout <<  Q[i*K+j]<<',';
			}
			cout<<endl;
		}
	}

  double loss=0; 

  //����ÿһ�ε�����ģ�loss��С��Ҳ����ԭ��R��������ÿһ����ȱʧֵ��Ԥ��ֵ��ƽ����ʧ 

  for(int i=0;i<N;++i) 

  { 

   for(int j=0;j<M;++j) 

   { 

    if(R[i*M+j]>0) 

    { 

     double error = 0; 

     for(int k=0;k<K;++k) 

      error += P[i*K+k]*Q[k*M+j]; 

     loss += pow(R[i*M+j]-error,2); 

     for(int k=0;k<K;++k) 

      loss += (beta/2) * (pow(P[i*K+k],2) + pow(Q[k*M+j],2)); 

    } 

   } 

  } 

  if(loss<0.001) {
  		cout<<"loss:"<<loss<<endl;
  	   break; 
  }

  if (step%1000==0) 

    cout<<"loss:"<<loss<<endl; 

 } 

} 

 

int main(int argc,char ** argv) 

{ 

 int N=5; //�û��� 

 int M=4; //��Ʒ�� 

 int K=2; //������� 

 double *R=new double[N*M]; 

 double *P=new double[N*K]; 

 double *Q=new double[M*K]; 

 R[0]=5,R[1]=3,R[2]=0,R[3]=1,R[4]=4,R[5]=0,R[6]=0,R[7]=1,R[8]=1,R[9]=1; 

 R[10]=0,R[11]=5,R[12]=1,R[13]=0,R[14]=0,R[15]=4,R[16]=0,R[17]=1,R[18]=5,R[19]=4; 

 

 cout<< "R����" << endl; 

 for(int i=0;i<N;++i) 

 { 

  for(int j=0;j<M;++j) 

   cout<< R[i*M+j]<<','; 

  cout<<endl; 

 } 

 

 //��ʼ��P��Q����������ˣ�ͨ��Ҳ���ԶԷ�����̬�ֲ������ݽ������������ 

 srand(1); 
cout << "-----"<<endl; 
 for(int i=0;i<N;++i) 
{
  	for(int j=0;j<K;++j) {
   		P[i*K+j]=rand()%9/float(10); 
//		P[i*K+j]=1;
   		cout <<  P[i*K+j]<<',';
	}
	cout<<endl;
}
 

 for(int i=0;i<K;++i) 

  for(int j=0;j<M;++j) 

   Q[i*M+j]=rand()%9/float(10); 
//	Q[i*M+j] = 1;

 cout <<"����ֽ� ��ʼ" << endl; 

 matrix_factorization(R,P,Q,N,M,K); 

 cout <<"����ֽ� ����" << endl; 

 

 cout<< "�ع�������R����" << endl; 

 for(int i=0;i<N;++i) 

 { 

  for(int j=0;j<M;++j) 

  { 

   double temp=0; 

   for (int k=0;k<K;++k) 

    temp+=P[i*K+k]*Q[k*M+j]; 

   cout<<temp<<','; 

  } 

  cout<<endl; 

 } 

 free(P),free(Q),free(R); 

 return 0; 

}
