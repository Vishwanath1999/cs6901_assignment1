import numpy as np
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist,mnist
from sklearn.model_selection import train_test_split
from scipy.special import log_softmax,softmax
import argparse
import seaborn as sns


def confusion_matrix(y_true, y_pred):

    classes = np.unique(y_true)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

    return cm

def precision_score(y_true, y_pred, average='binary'):

    classes = np.unique(y_true)
    n_classes = len(classes)
    precision = np.zeros(n_classes)

    for i in range(n_classes):
        true_positive = np.sum((y_true == classes[i]) & (y_pred == classes[i]))
        false_positive = np.sum((y_true != classes[i]) & (y_pred == classes[i]))
        precision[i] = true_positive / (true_positive + false_positive)
    if average == 'binary':
        return precision[1]  
    elif average == 'micro':
        true_positive = np.sum(y_true == y_pred)
        false_positive = np.sum(y_true != y_pred)
        return true_positive / (true_positive + false_positive)
    elif average == 'macro':
        return np.mean(precision)
    elif average == 'weighted':
        class_count = np.bincount(y_true)
        weights = class_count / np.sum(class_count)
        return np.average(precision, weights=weights)
    else:
        raise ValueError("The 'average' parameter must be one of 'binary', 'micro', 'macro', or 'weighted'.")

class FFNN:
    def __init__(self,net_size,layer_act,init_wb='random',lr=1e-3,opt='rmsprop',lamda=0,batch_size=64,\
                 n_epochs=10,gamma=0.9,beta=0.999,beta_1=0.9,beta_2=0.999,seed=None,loss='cross_ent',relu_param=0,epsilon=1e-8):
        
        self.net_size = net_size
        self.layer_acts = layer_act
        self.init_wb = init_wb
        self.lr = lr
        self.optim = opt
        self.lamda = lamda
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss = loss
        self.seed = seed
        self.relu_param=relu_param
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta

    def onehot_encode(self,y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.T
    
    def nn_init(self, network_size, wb_init='random'):

        if self.seed is not None:
            np.random.seed(self.seed)

        params = {}

        num_layers = len(network_size)

        if wb_init == 'random':
            for layer in range(1, num_layers):
                params['weights' + str(layer)] = np.random.normal(0,1,(network_size[layer], network_size[layer - 1]))
                params['biases' + str(layer)] = np.random.normal(0,1,(network_size[layer], 1))
                
        elif wb_init == 'xavier_uniform':
            for layer in range(1, num_layers):
                r = np.sqrt(6.0 / (network_size[layer] + network_size[layer - 1]))
                params['weights' + str(layer)] = np.random.uniform(-r, r, (network_size[layer], network_size[layer - 1]))
                params['biases' + str(layer)] = np.random.uniform(-r, r, (network_size[layer], 1))
        
        else:
            raise ValueError('Invalid Activation function ...')
        return params
    
    def Linear(self,input_data,diff=False):
        input_data = np.array(input_data, dtype=np.float64)
        if diff == False:
            return input_data
        else:
            return np.ones_like(input_data)

    def ReLU(self, input_data, diff=False):
        alpha = self.relu_param
        input_data = np.array(input_data, dtype=np.float64)

        if diff == False:
            return np.where(input_data < 0, alpha * input_data, input_data)

        elif diff == True:
            output_data = np.ones_like(input_data, dtype=np.float64)
            output_data[input_data < 0] = alpha
            return output_data
    
    # def ELU(self,input_data,diff=False):
    #     alpha = self.relu_param
    #     input_data = np.where(input_data>700,700,input_data)
    #     if diff == False:
    #         return np.where(input_data < 0, alpha * (np.exp(input_data)-1), input_data)
    #     else:
    #         output_data = np.ones_like(input_data, dtype=np.float64)
    #         output_data[input_data < 0] = alpha*np.exp(input_data)
    #         return output_data  
    
    def sigmoid(self, input_data, diff=False):
        input_data = np.where(input_data<-700,-700,input_data)
        if not diff:
            output_data = 1 / (1 + np.exp(-np.array(input_data)))
        else:
            s = 1 / (1 + np.exp(-np.array(input_data)))
            output_data = s * (1 - s)
        return output_data

    def Tanh(self, input_data, diff=False):
        input_data = np.array(input_data)
        input_data = np.clip(input_data,-700,700)
        if not diff:
            output_data = np.tanh(input_data)
        else:
            output_data = 1 - np.tanh(input_data) ** 2
        return output_data
    
    def softmax(self,X):
        X = np.clip(X,-700,700)
        return log_softmax(X,axis=0)

    def forward(self,data,acts,params):
        if self.seed is not None:
            np.random.seed(self.seed)
        param_list = []
        act_out = data
        for idx, act in enumerate(acts,start=1):
            data_prev = act_out
            Wb = np.dot(params['weights'+str(idx)],data_prev)+params['biases'+str(idx)]

            if act == 'sigmoid':
                act_out = self.sigmoid(Wb)
            elif act == 'tanh':
                act_out = self.Tanh(Wb)
            elif act == 'relu':
                act_out = self.ReLU(Wb)
            elif act == 'softmax':
                act_out = self.softmax(Wb)
            elif act == 'identity':
                act_out = self.Linear(Wb)
            # elif act == 'elu':
            #     act_out == self.ELU(Wb)
            else:
                raise ValueError('Invalid activation function ...')
            
            pl = ((data_prev,params['weights'+str(idx)],params['biases'+str(idx)]),Wb)
            param_list.append(pl)
        return act_out,param_list
    
    def grad(self,pred,target,params,lamda=0,loss='cross_ent'):
        n_class = target.shape[1]
        if loss == 'cross_ent':
            loss = -np.mean(np.multiply(pred,target),axis=1).sum()
        elif loss=='mse':
            loss = -np.mean(np.multiply(pred-target,pred-target),axis=1).sum()
        else:
            raise ValueError('Error function invalid. Please choose either "cross_ent" or "mse" ')
        param_len = len(params)//2

        sum_w = 0
        for idx in range(1,param_len):
            sum_w += np.square(params['weights'+str(idx)]).sum()
        loss += sum_w*(lamda/(2*n_class))
        return loss

    def backward(self,pred,target,param_list,acts,lamda=0,loss='cross_ent'):
        grad_tape = {}
        lpl = len(param_list)
        m,n = pred.shape
        target = target.reshape(pred.shape)
        if loss == 'cross_ent':
            dOut = np.exp(pred) - target
        elif loss == 'mse':
            dOut = 2*(np.exp(pred) - target)
        else:
            raise ValueError('Error function invalid. Please choose either "cross_ent" or "mse" ')

        pred,weight,_ = param_list[-1][0]
        grad_tape['d_weights'+str(lpl)] = np.dot(dOut,pred.T)/m
        grad_tape['d_biases'+str(lpl)] = dOut.sum(axis=1,keepdims=True)/m
        grad_tape['d_pred'+str(lpl-1)] = np.dot(weight.T,dOut)

        for idx in reversed(range(lpl-1)):
            linear_pred,out = param_list[idx]
            out_prev,weight,b = linear_pred

            m,n = out_prev.shape
            dOut_prev = grad_tape['d_pred'+str(idx+1)]

            if acts[idx] == 'relu':
                dOut = dOut_prev*self.ReLU(out,True)
            elif acts[idx] == 'sigmoid':
                dOut = dOut_prev*self.sigmoid(out,True)
            elif acts[idx] == 'tanh':
                dOut = dOut_prev*self.Tanh(out,True)
            # elif acts[idx] == 'elu':
            #     dOut = dOut_prev*self.ELU(out,True)
            elif acts[idx] == 'identity':
                dOut = dOut_prev*self.Linear(out,True)
            else:
                raise ValueError('Enter a valid activation ..')          
            
            grad_tape['d_pred'+str(idx)] = np.dot(weight.T,dOut)
            grad_tape['d_weights'+str(idx+1)] = (np.dot(dOut,out_prev.T)+ lamda*weight)/m
            grad_tape['d_biases'+str(idx+1)] = dOut.sum(axis=1,keepdims=True)/m
        return grad_tape
        

    def optim_step(self,params,grad_tape,lr,t_step,algo='adam',moment_params=None):
        len_param = len(params)//2

        if algo == 'sgd':
            for idx in range(len_param):
                params['weights'+str(idx+1)] -= lr*grad_tape['d_weights'+str(idx+1)]
                params['biases'+str(idx+1)] -= lr*grad_tape['d_biases'+str(idx+1)]
                moment_params=None
        elif algo == 'sgdm':
            for idx in range(len_param):
                moment_params['v_w'+str(idx+1)] = self.gamma*moment_params['v_w'+str(idx+1)] + (1-self.gamma)*grad_tape['d_weights'+str(idx+1)]
                moment_params['v_b'+str(idx+1)] = self.gamma*moment_params['v_b'+str(idx+1)] + (1-self.gamma)*grad_tape['d_biases'+str(idx+1)]

                params['weights'+str(idx+1)] -= lr*moment_params['v_w'+str(idx+1)]
                params['biases'+str(idx+1)] -= lr*moment_params['v_b'+str(idx+1)]
        elif algo == 'nag':
            for idx in range(len_param):
                moment_params['v_w'+str(idx+1)] = self.gamma*moment_params['v_w'+str(idx+1)] - lr*grad_tape['d_weights'+str(idx+1)]
                moment_params['v_b'+str(idx+1)] = self.gamma*moment_params['v_b'+str(idx+1)] - lr*grad_tape['d_biases'+str(idx+1)]

                params['weights'+str(idx+1)] -= self.gamma*(moment_params['v_w'+str(idx+1)] - moment_params['v_w_prev'+str(idx+1)])
                params['biases'+str(idx+1)] -= self.gamma*(moment_params['v_b'+str(idx+1)] - moment_params['v_b_prev'+str(idx+1)])

                moment_params['v_w_prev'+str(idx+1)] = moment_params['v_w'+str(idx+1)]
                moment_params['v_b_prev'+str(idx+1)] = moment_params['v_b'+str(idx+1)]

        elif algo == 'rmsprop':
            for idx in range(len_param):
                moment_params['m_b'+str(idx+1)] = self.beta*moment_params['m_b'+str(idx+1)] + (1-self.beta)*(grad_tape['d_biases'+str(idx+1)]**2)
                moment_params['m_w'+str(idx+1)] = self.beta*moment_params['m_w'+str(idx+1)] + (1-self.beta)*(grad_tape['d_weights'+str(idx+1)]**2)

                params['weights'+str(idx+1)] -= lr*grad_tape['d_weights'+str(idx+1)]/(np.sqrt(moment_params['m_w'+str(idx+1)])+1e-8)
                params['biases'+str(idx+1)] -= lr*grad_tape['d_biases'+str(idx+1)]/(np.sqrt(moment_params['m_b'+str(idx+1)])+1e-8)

        elif algo == 'adam':
            for idx in range(len_param):
                moment_params['v_b'+str(idx+1)] = self.beta_1*moment_params['v_b'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_biases'+str(idx+1)]
                moment_params['v_w'+str(idx+1)] = self.beta_1*moment_params['v_w'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_weights'+str(idx+1)]

                moment_params['m_b'+str(idx+1)] = self.beta_2*moment_params['m_b'+str(idx+1)] + (1-self.beta_2)*(grad_tape['d_biases'+str(idx+1)]**2)
                moment_params['m_w'+str(idx+1)] = self.beta_2*moment_params['m_w'+str(idx+1)] + (1-self.beta_2)*(grad_tape['d_weights'+str(idx+1)]**2)

                mod_lr = lr*np.sqrt((1-self.beta_2**t_step)/(1-self.beta_1**t_step+1e-8))
                params['weights'+str(idx+1)] -= mod_lr*(moment_params['v_w'+str(idx+1)]/(np.sqrt(moment_params['m_w'+str(idx+1)])+1e-8))
                params['biases'+str(idx+1)] -= mod_lr*(moment_params['v_b'+str(idx+1)]/(np.sqrt(moment_params['m_b'+str(idx+1)])+1e-8))
        elif algo =='nadam':
            for idx in range(len_param):
                moment_params['v_b'+str(idx+1)] = self.beta_1*moment_params['v_b'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_biases'+str(idx+1)]
                moment_params['v_w'+str(idx+1)] = self.beta_1*moment_params['v_w'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_weights'+str(idx+1)]

                moment_params['m_b'+str(idx+1)] = self.beta_2*moment_params['m_b'+str(idx+1)] + (1-self.beta_2)*(grad_tape['d_biases'+str(idx+1)]**2)
                moment_params['m_w'+str(idx+1)] = self.beta_2*moment_params['m_w'+str(idx+1)] + (1-self.beta_2)*(grad_tape['d_weights'+str(idx+1)]**2)

                mod_lr = lr*np.sqrt((1-self.beta_2**t_step)/(1-self.beta_1**t_step+1e-8))
                params['weights'+str(idx+1)] -= (mod_lr/(np.sqrt(moment_params['m_w'+str(idx+1)])+1e-8))*(self.beta_1*moment_params['v_w'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_weights'+str(idx+1)])
                params['biases'+str(idx+1)] -= (mod_lr/(np.sqrt(moment_params['m_b'+str(idx+1)])+1e-8))*(self.beta_1*moment_params['v_b'+str(idx+1)] + (1-self.beta_1)*grad_tape['d_biases'+str(idx+1)])
        return params,moment_params
    
    def predict(self,data):
        out = self.forward(data,self.layer_acts,self.params)[0]
        return np.argmax(out,axis=0),out.T
    
    def train(self,X_train,Y_train,X_val,Y_val,n_classes=10,wb_log=True):
        self.losses=[]
        moment_params = {}
        m = X_train.shape[1]
        y_train = self.onehot_encode(Y_train,n_classes)
        y_val = self.onehot_encode(Y_val,n_classes)
        self.params = self.nn_init(self.net_size,self.init_wb)
        self.t_step = 1
        idx = np.arange(m)

        if self.optim != 'sgd':
            for ii in range(1,len(self.net_size)):
                moment_params['v_w'+str(ii)] = np.zeros((self.net_size[ii],self.net_size[ii-1]))
                moment_params['v_b'+str(ii)] = np.zeros((self.net_size[ii],1))

                moment_params['v_w_prev'+str(ii)] = np.zeros((self.net_size[ii],self.net_size[ii-1]))
                moment_params['v_b_prev'+str(ii)] = np.zeros((self.net_size[ii],1))

                moment_params['m_w'+str(ii)] = np.zeros((self.net_size[ii],self.net_size[ii-1]))
                moment_params['m_b'+str(ii)] = np.zeros((self.net_size[ii],1))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            X_shuffled = X_train[:,idx]
            Y_shuffled = y_train[:,idx]
            for ii in range(0,m,self.batch_size):
                X_batched = X_shuffled[:,ii:ii+self.batch_size]
                Y_batched = Y_shuffled[:,ii:ii+self.batch_size]

                out,param_list = self.forward(X_batched,self.layer_acts,self.params)                
                loss = self.grad(out,Y_batched,self.params,self.lamda,self.loss)                
                self.losses.append(loss)
                grads = self.backward(out,Y_batched,param_list,self.layer_acts,self.lamda,self.loss)
                self.params,moment_params = self.optim_step(self.params,grads,self.lr,\
                                                            self.t_step,self.optim,moment_params)
                self.t_step+=1
            y_pred_train,_ = self.predict(X_train)
            y_pred_valid,_ = self.predict(X_val)

            train_acc = accuracy_score(Y_train,y_pred_train)
            val_acc = accuracy_score(Y_val,y_pred_valid)
            val_out,_ = self.forward(X_val,self.layer_acts,self.params)
            val_loss = self.grad(val_out,y_val,self.params,self.lamda,self.loss)
            log = {'train_acc':train_acc, 'val_acc':val_acc,'train_loss':loss,'val_loss':val_loss}#
            if wb_log:
                wandb.log(log)
            else:
                print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset',type=str,default='fashion_mnist',choices=['fashion_mnist','mnist'],help='Dataset choice')
    parser.add_argument('-e','--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('-nhl','--num_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('-sz','--hidden_size', type=int, default=128, help='Number of hidden units per layer')
    parser.add_argument('-w_d','--weight_decay', type=float, default=0.0, help='L2 regularization coefficient')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-o','--optimizer', type=str, default='nadam', choices=['adam', 'sgd','sgdm','nag','nadam'], help='Optimizer')
    parser.add_argument('-b','--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-w_i','--weight_init', type=str, default='xavier_normal', choices=['xavier', 'random'], help='Weight initialization method')
    parser.add_argument('-a','--activation', type=str, default='sigmoid', choices=['relu', 'sigmoid','tanh','identity'], help='Activation function')
    parser.add_argument('-l','--loss', type=str, default='mse', choices=['mse', 'cross_ent'], help='Loss function')
    parser.add_argument('-r_p','--relu_param', type=float, default=0, help='ReLU parameter')
    parser.add_argument('-we','--wandb_entity',type=str,default='name',help='Name of Wandb entity')
    parser.add_argument('-wp','--wandb_project',type=str,default='project', help='Project Name')
    parser.add_argument('-m','--momentum',type=float,default=0.9,help='Momentum for sgdm and nag')
    parser.add_argument('-beta','--beta',type=float,default=0.999,help='For RMSProp')
    parser.add_argument('-eps','--epsilon',type=float,default=1e-8,help='Epsilon for optimizers')
    parser.add_argument('-beta1','--beta1',type=float,default=0.9,help='Beta1 used by adam and nadam optimizers')
    parser.add_argument('-beta2','--beta2',type=float,default=0.999,help='Beta2 used by adam and nadam optimizers')
    args = parser.parse_args()
    config = vars(args)
    wandb.init(config=config,entity=args.wandb_entity, project=args.wandb_project)

    if args.dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise ValueError('Choose from mnist or fashion_mnist ...')
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

    x_train = x_train.reshape((len(x_train), 28*28))
    x_train = x_train.astype('float32') / 255

    x_valid = x_valid.reshape((len(x_valid), 28*28))
    x_valid = x_valid.astype('float32') / 255

    # Preprocessing test data
    x_test = x_test.reshape((len(x_test), 28 * 28))
    x_test = x_test.astype('float32') / 225

    X = x_train.T
    X_valid = x_valid.T
    Y_valid = y_valid
    Y = y_train
    n_class= 10

    layers = []
    for i in range(args.num_layers+2):
        if i == 0:
            layers.append(X.shape[0])
        elif i == args.n_hidden+1:
            layers.append(n_class)
        else:
            layers.append(args.hidden_size)
        i = i+1
    act = []
    o_act = 'softmax'
    for i in range(args.num_layers+1):
        if i == args.num_layers:
            act.append(o_act)
        else:
            act.append(args.activation)
        i = i+1
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model = FFNN(net_size=layers,layer_act=act,init_wb=args.weight_init,lr=args.learning_rate,opt=args.optimizer,\
                 lamda=args.weight_decay,batch_size=args.batch_size,n_epochs=args.epochs,loss=args.loss,\
                    relu_param=args.relu_param,gamma=args.momentum,beta=args.beta,epsilon=args.epsilon,beta_1=args.beta1,beta2=args.beta2)
    model.train(X,Y,X_valid,Y_valid)
    y_test_pred,_ = model.predict(x_test.T)

    cm = confusion_matrix(y_test, y_test_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, cmap="Blues", square=True, xticklabels=class_names, yticklabels=class_names
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # Log the confusion matrix plot to wandb
    wandb.log({"Confusion Matrix":wandb.Image(fig)})
    fig.clf()
    plt.close('all')

