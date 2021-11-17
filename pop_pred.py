# convert pp_rec's popularity predictor to pytorch version
# https://github.com/taoqi98/PP-Rec/blob/main/Code/Encoders.py
# https://aclanthology.org/2021.acl-long.424.pdf

class pop_pred(nn.Module):
    def __init__(self, input_dim, npratio):
        self.linear1 = nn.Linear(400, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128,1, bias=False)

        self.linear5 = nn.Linear(input_dim - 400, 64)
        self.linear6 = nn.Linear(64,64)
        self.linear7 = nn.Linear(64,1, bias=False)

        self.linear8 = nn.Linear(input_dim, 128)
        self.linear9 = nn.Linear(128, 64)
        self.linear10 = nn.Linear(64,1)

        self.scaler = nn.Linear(1 + npratio, 1, bias=False)
        nn.init.constant_(self.scaler.weight, 19)

        self.w_c = torch.nn.Parameter(torch.rand(1,1))
        self.w_p = torch.nn.Parameter(torch.rand(1,1))


        self.npratio = npratio
        
        self.criterion = nn.Catego

    def forward(self, clicked_news_vecs, candidates_ctr, compute_loss=True):
        x1= clicked_news_vecs[:, :400]
        x2 = clicked_news_vecs[:, 400:]

        x1 = torch.tanh(self.linear1(x1))
        x1 = torch.tanh(self.linear2(x1))
        x1 = self.linear3(x1)
        bias_content_score = self.linear4(x1)

        x2 = torch.tanh(self.linear5(x2))
        x2 = torch.tanh(self.linear6(x2))
        bias_recency_score = self.linear7(x)

        gate = torch.tanh(self.linear8(x))
        gate = torch.tanh(self.linear9(gate))
        gate = torch.sigmoid(self.linear10(gate))

        bias_content_score = (1 - gate) * bias_content_score + gate * bias_recency_score

        ctrs = candidates_ctr.reshape((-1, 1))
        bias_ctr_score = scaler(ctrs).reshape(-(1,))

        s_p = self.w_c * bias_ctr_score + self.w_p * bias_content_score
        # if compute_loss:
        #     nn.NLLLoss()(torch.log(s_p), )

        return s_p

        


    


# tf code


# bias_content_vec = Input(shape=(500,))
# vec1 = keras.layers.Lambda(lambda x:x[:,:400])(bias_content_vec)
# vec2 = keras.layers.Lambda(lambda x:x[:,400:])(bias_content_vec)

# vec1 = Dense(256,activation='tanh')(vec1)
# vec1 = Dense(256,activation='tanh')(vec1)
# vec1 = Dense(128,)(vec1)
# bias_content_score = Dense(1,use_bias=False)(vec1)

# vec2 = Dense(64,activation='tanh')(vec2)
# vec2 = Dense(64,activation='tanh')(vec2)
# bias_recency_score = Dense(1,use_bias=False)(vec2)

# gate = Dense(128,activation='tanh')(bias_content_vec)
# gate = Dense(64,activation='tanh')(gate)
# gate = Dense(1,activation='sigmoid')(gate)

# bias_content_score = keras.layers.Lambda(lambda x: (1-x[0])*x[1]+x[0]*x[2] )([gate,bias_content_score,bias_recency_score])

# bias_content_scorer = Model(bias_content_vec,bias_content_score)

# scaler =  Dense(1,use_bias=False,kernel_initializer=keras.initializers.Constant(value=19))
# candidates_ctr = keras.Input((1+config['npratio'],), dtype='float32')
# ctrs = keras.layers.Reshape((1+config['npratio'],1))(candidates_ctr)
# ctrs = scaler(ctrs)
# bias_ctr_score = keras.layers.Reshape((1+config['npratio'],))(ctrs)