class Semilabel(nn.Module):
    def __init__(self, input_dim=args.input_dim, hidden_dim=256, feature_dim=256, output_dim=20):
        super().__init__()

        self.input = nn.Linear(input_dim, hidden_dim//2 )
        self.ln1 = nn.LayerNorm(hidden_dim//2)

        self.hidden1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, bidirectional=True)
        self.ln3 = nn.LayerNorm(2 * hidden_dim)

        self.fc0 = nn.Linear(2 * hidden_dim, 1)

        self.hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)

        self.predict = nn.Linear(hidden_dim, len(dic))



    def forward(self, arrays):
        hidden_states = swish_fn(self.ln1(self.input(arrays)))
        hidden_states = swish_fn(self.ln2(self.hidden1(hidden_states)))
        hidden_states, _ = self.lstm(hidden_states.view(len(hidden_states), 1, -1))
        # hidden_states_ = hidden_states.transpose(1, 0)
        # hidden_states1 = hidden_states_.transpose(2, 1)
        # m = nn.MaxPool1d(hidden_states1.shape[2])
        # output = m(hidden_states1).reshape((hidden_states1.shape[0], 1, hidden_states1.shape[1]))
        # similarity = torch.cosine_similarity(hidden_states_, output, dim=2).reshape(hidden_states_.shape[0],
        #                                                                             hidden_states_.shape[1], 1)
        # hidden_states = torch.mul(hidden_states_, similarity)
        # hidden_states = hidden_states.squeeze(0)
        hidden_states = hidden_states.squeeze(1)
        align = F.softmax(self.fc0(hidden_states), dim=0)

        hidden_states = torch.mul(hidden_states, align)

        hidden_states = swish_fn(self.ln3(hidden_states))
        hidden_states = swish_fn(self.ln4(self.hidden(hidden_states)))
        output = self.predict(hidden_states)
        # output = F.softmax(hidden_states, dim=0)
        # output = self.dropout(hidden_states)

        return output
