class FST:
    def __init__(self):
        self.arcs = dict()
        self.arcs_from_state_sym_in = dict()
        self.arcs_from_state_sym_out = dict()
        self.final_states = set()

    def add_final(self, state):
        self.final_states.add(state)

    def add_arc(self, state_from, state_to, sym_in, sym_out, prob):
        key = (state_from, state_to, sym_in, sym_out)
        # print(f'Arc {key} -> {prob}')
        self.arcs[key] = prob

        if not (state_from, sym_in) in self.arcs_from_state_sym_in:
            self.arcs_from_state_sym_in[state_from, sym_in] = list()
        self.arcs_from_state_sym_in[state_from, sym_in].append( (state_to, sym_out, prob) )

        if not (state_from, sym_out) in self.arcs_from_state_sym_out:
            self.arcs_from_state_sym_out[state_from, sym_out] = list()
        self.arcs_from_state_sym_out[state_from, sym_out].append( (state_to, sym_in, prob) )

    def translate(self, state_from, sym_in):
        if (state_from, sym_in) in self.arcs_from_state_sym_in:
            return self.arcs_from_state_sym_in[state_from, sym_in]
        else:
            return []

    def translate_reverse(self, state_from, sym_out):
        if (state_from, sym_out) in self.arcs_from_state_sym_out:
            return self.arcs_from_state_sym_out[state_from, sym_out]
        else:
            return []