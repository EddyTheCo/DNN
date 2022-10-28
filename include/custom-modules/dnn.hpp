#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>
#ifdef USE_YAML
#include<yaml-cpp/yaml.h>
#endif


namespace custom_models{
class DNNImpl : public torch::nn::Module {
    public:
    DNNImpl(const std::vector<int64_t>& layers) {

          for(auto i=0;i<layers.size()-1;i++)
          {
              auto var=torch::nn::Linear(torch::nn::LinearOptions(layers[i],layers[i+1]).bias(false));
		
              var->weight.data()=2*torch::rand({layers[i+1],layers[i]})-1; // initialize uniformly the parameters of the model

              Layers.push_back(register_module("linear_"+std::to_string(i),var));

          }

      }
    void update(void)const{};
#ifdef USE_YAML
            DNNImpl(YAML::Node config):DNNImpl(config["Layers"].as<std::vector<int64_t>>()){std::cout<<config<<std::endl;};
#endif
      torch::Tensor forward(at::Tensor x ) {

          int i;
          for( i=0;i<Layers.size()-1;i++)
          {
                x=torch::nn::functional::leaky_relu(Layers[i](x));
          }

          x=torch::nn::functional::softmax(torch::tanh(Layers[i](x)),
                  torch::nn::functional::SoftmaxFuncOptions(-1));

          return x;
        }
      std::vector<torch::nn::Linear> Layers;
};
TORCH_MODULE(DNN);
};
