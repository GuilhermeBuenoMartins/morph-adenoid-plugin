package neural.network;

import neural.network.core.FNArray;

public interface Layer {

    void load(FNArray w, FNArray b);

    FNArray predict(FNArray x);
    
}
