
def GD(X,y,model,epochs=20, lr=0.01):
    history=[]
    for epoch in range(epochs):
        y_pred = [model(x) for x in X]
        loss = sum((y[i]-y_pred[i])**2 for i in range(len(y)))
        
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data-=lr*p.grad
            
        history.append(loss.data)    
            
        print(f'Epoch {epoch+1}: Loss = {loss.data}')
    
    return history    


def accuracy(X,y, model):
    correct = sum((model(xi).data > 0 ) == (yi>0) for xi, yi in zip(X,y))        
    return correct/len(y)            