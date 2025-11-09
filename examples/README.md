# SEU Injection Framework - Examples

This directory contains comprehensive examples demonstrating the SEU Injection Framework for various research applications in harsh environment neural network deployment.

## 🚀 Quick Start

All examples are self-contained and can be run directly:

```bash
# Install the framework
pip install seu-injection-framework

# Run basic CNN robustness analysis
python basic_cnn_robustness.py

# Run architecture comparison study  
python architecture_comparison.py
```

## 📋 Example Overview

### **📓 Interactive Jupyter Notebook**
- **[Example_Attack_Notebook.ipynb](Example_Attack_Notebook.ipynb)** - Comprehensive interactive tutorial covering all framework capabilities with visualizations

### **🔬 Python Scripts**

### 1. `basic_cnn_robustness.py` - Space Mission CNN Analysis

**Research Focus**: Mars rover image classification robustness assessment

**Key Features**:
- ✅ **Layer vulnerability analysis** - Identify most fault-sensitive layers
- ✅ **Bit position sensitivity** - IEEE 754 fault impact assessment  
- ✅ **Stochastic SEU campaigns** - Realistic space radiation simulation
- ✅ **Mission readiness reports** - Professional assessment documentation

**Applications**:
- 🛰️ Space mission neural network validation
- ☢️ Nuclear facility AI system assessment
- ✈️ Aviation safety-critical system evaluation
- 🏭 Industrial harsh environment deployment

**Output**:
- Comprehensive vulnerability analysis across layer types
- Bit-level sensitivity mapping for IEEE 754 representation
- Statistical robustness assessment for different radiation environments
- Professional mission readiness report (`mission_robustness_report.txt`)

**Runtime**: ~2-3 minutes on CPU, ~30 seconds on GPU

---

### 2. `architecture_comparison.py` - Systematic Architecture Evaluation

**Research Focus**: Comparative robustness analysis across neural network architectures

**Key Features**:
- 🏗️ **Multi-architecture testing** - SimpleNN, CompactCNN, MiniResNet, EfficientNet
- 📊 **Complexity vs robustness analysis** - Parameter count vs fault tolerance
- 📈 **Comprehensive benchmarking** - Bit sensitivity, layer vulnerability, stochastic robustness
- 📋 **Selection guidance** - Mission-specific architecture recommendations

**Applications**:
- 🎯 Architecture selection for critical deployments
- 🔬 Research into fault-tolerant neural network design
- 🛡️ Defense system AI component evaluation
- 🏥 Medical device neural network validation

**Output**:
- Robustness ranking across all tested architectures
- Detailed vulnerability analysis for each model type
- Complexity-robustness trade-off visualizations (`architecture_comparison.png`)
- Comprehensive deployment recommendations (`architecture_comparison_report.txt`)

**Runtime**: ~5-7 minutes on CPU, ~1-2 minutes on GPU

---

## 🔬 Research Methodology

### SEU Injection Strategy

All examples implement the research methodology from:
**"A Framework for Developing Robust Machine Learning Models in Harsh Environments"**

**Key Principles**:
1. **IEEE 754 Compliance** - Precise bit-level fault injection in floating-point weights
2. **Statistical Significance** - Multiple trial averaging with confidence intervals
3. **Layer-Specific Analysis** - Targeted vulnerability assessment across model components
4. **Realistic Scenarios** - Stochastic injection matching actual radiation environments

### Experimental Design

#### **Deterministic SEU Injection**
- **Purpose**: Precise vulnerability mapping
- **Method**: Single bit flip at specified position in target layer
- **Applications**: Critical layer identification, worst-case analysis

#### **Stochastic SEU Injection** 
- **Purpose**: Realistic radiation simulation
- **Method**: Probabilistic bit flips across multiple parameters
- **Applications**: Mission risk assessment, statistical robustness analysis

#### **Cross-Architecture Comparison**
- **Purpose**: Architecture selection guidance
- **Method**: Standardized test protocol across model types
- **Applications**: System design, deployment planning

## 📊 Interpreting Results

### Vulnerability Metrics

#### **Accuracy Drop** (Primary Metric)
- **Range**: 0.0 (no impact) to 1.0 (complete failure)
- **Interpretation**:
  - `< 0.01`: 🟢 **Low risk** - Acceptable for most missions
  - `0.01 - 0.05`: 🟡 **Moderate risk** - Consider protection measures
  - `0.05 - 0.1`: 🟠 **High risk** - Requires hardening
  - `> 0.1`: 🔴 **Critical risk** - Not suitable without extensive protection

#### **Layer Vulnerability Analysis**
- **Convolutional layers**: Often robust due to spatial redundancy
- **Fully connected layers**: Typically most vulnerable, especially final classifier
- **Batch normalization**: Can amplify or mitigate SEU effects
- **Residual connections**: May provide natural error correction

#### **Bit Position Sensitivity**
- **Mantissa bits (0-22)**: Variable impact, position-dependent
- **Exponent bits (23-30)**: Often catastrophic due to magnitude changes
- **Sign bit (31)**: Can cause systematic prediction inversions

### Risk Assessment Guidelines

#### **Space Mission Classification**
- **Low Earth Orbit (LEO)**: Moderate radiation, p~1e-6
- **Deep Space Transit**: High radiation, p~1e-5  
- **Jupiter/Saturn Missions**: Extreme radiation, p~1e-4
- **Solar Storm Events**: Crisis scenarios, p~1e-3

#### **Protection Strategies by Risk Level**
- **🟢 Low Risk**: Standard deployment, monitoring recommended
- **🟡 Moderate Risk**: Error detection, periodic validation
- **🟠 High Risk**: Triple Modular Redundancy (TMR), error correction
- **🔴 Critical Risk**: Radiation-hardened hardware required

## 🛠️ Customization Guide

### Adapting for Your Research

#### **1. Custom Neural Network Architectures**

```python
# Add your model to the architecture comparison
class YourCustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super(YourCustomModel, self).__init__()
        # Your architecture here
        
    def forward(self, x):
        # Your forward pass
        return x

# Add to comparison
architectures['YourModel'] = YourCustomModel()
```

#### **2. Domain-Specific Data**

```python
# Replace synthetic data with your domain data
def load_your_mission_data():
    # Load actual satellite imagery, sensor data, etc.
    X = load_mission_images()
    y = load_mission_labels() 
    return X, y

# Use in analysis
X, y = load_your_mission_data()
dataset = TensorDataset(X, y)
```

#### **3. Custom Robustness Metrics**

```python
# Define domain-specific metrics
def custom_mission_metric(model, X, y, device=None):
    """Custom metric for your specific application."""
    with torch.no_grad():
        outputs = model(X.to(device))
        # Your evaluation logic here
        return custom_score

# Use in SEU injection
injector.run_seu(
    data=data_loader,
    criterion=custom_mission_metric,
    # ... other parameters
)
```

#### **4. Mission-Specific Scenarios**

```python
# Define your radiation environment
mission_scenarios = {
    'Your Mission': {
        'probability': 2e-5,  # Your environment's SEU rate
        'description': 'Your mission description'
    }
}
```

### Performance Optimization Tips

#### **For Large Models**
- Use GPU acceleration: `device='cuda'`
- Reduce dataset size for initial testing
- Target specific critical layers rather than full model analysis

#### **For Extensive Studies**
- Implement parallel processing across bit positions
- Cache baseline accuracy calculations
- Use deterministic random seeds for reproducibility

#### **For Production Deployment**
- Implement continuous monitoring of selected vulnerability metrics
- Set up automated alerts for degradation beyond acceptable thresholds
- Maintain fallback models with higher robustness ratings

## 📚 Extended Research Applications

### Academic Research Integration

#### **Reproducible Research Protocol**
1. **Environment Documentation**: Record exact framework version, dependencies
2. **Seed Management**: Use consistent random seeds across experiments  
3. **Statistical Validation**: Report confidence intervals, significance tests
4. **Methodology Description**: Reference framework paper and configuration

#### **Publication Guidelines**
- **Citation**: Include framework citation in methodology section
- **Results Reporting**: Use standardized vulnerability metrics
- **Code Sharing**: Share example configurations for reproducibility
- **Benchmark Contribution**: Consider contributing results to community benchmarks

### Industry Applications

#### **Certification Support**
- Generate compliance reports for safety standards (DO-178C, IEC 61508)
- Document systematic verification and validation processes
- Provide quantitative robustness evidence for regulatory approval

#### **System Integration**
- Interface with existing ML deployment pipelines
- Integrate with continuous integration/deployment (CI/CD) workflows
- Develop automated robustness regression testing

## 🔗 Additional Resources

### Framework Documentation
- **Main Repository**: https://github.com/wd7512/seu-injection-framework
- **Installation Guide**: ../docs/installation.md
- **API Documentation**: ../docs/api/
- **Contributing Guidelines**: ../CONTRIBUTING.md

### Research Background
- **Methodology Paper**: Framework research publication
- **SEU Physics**: IEEE 754 standard, radiation effects literature
- **Fault Tolerance**: Neural network robustness research

### Community Support
- **GitHub Issues**: Bug reports, feature requests
- **Research Questions**: Use research issue template
- **Direct Contact**: wd7512@bristol.ac.uk

---

## 📞 Support & Questions

### Getting Help
1. **Check Documentation**: Review README.md and framework documentation
2. **Search Issues**: Look for similar questions in GitHub issues
3. **Create Issue**: Use appropriate template (bug/feature/research)
4. **Direct Contact**: Email for research collaboration or urgent issues

### Contributing Examples
We welcome community contributions of additional examples:
- **Novel applications**: New domains or use cases
- **Advanced techniques**: Sophisticated analysis methodologies  
- **Performance optimizations**: Faster or more comprehensive approaches
- **Validation studies**: Real-world deployment case studies

See CONTRIBUTING.md for contribution guidelines and quality standards.

---

*These examples demonstrate production-ready usage of the SEU Injection Framework for systematic neural network robustness analysis in harsh environments.*