ъЯ
ч
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-rc1-8-g6887368d6d48ѓ	
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Є
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_5/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:*
dtype0
Є
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_5/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:*
dtype0
Ў
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_5/kernel/*
dtype0*
shape
:*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes

:*
dtype0
Ў
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_5/kernel/*
dtype0*
shape
:*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes

:*
dtype0
Є
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_4/bias/*
dtype0*
shape:0*$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
:0*
dtype0
Є
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_4/bias/*
dtype0*
shape:0*$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
:0*
dtype0
Ў
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_4/kernel/*
dtype0*
shape
:@0*&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes

:@0*
dtype0
Ў
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_4/kernel/*
dtype0*
shape
:@0*&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes

:@0*
dtype0
Ї
Adam/v/conv2d_5/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv2d_5/bias/*
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_5/bias
y
(Adam/v/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/bias*
_output_shapes
:@*
dtype0
Ї
Adam/m/conv2d_5/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv2d_5/bias/*
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_5/bias
y
(Adam/m/conv2d_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/bias*
_output_shapes
:@*
dtype0
Й
Adam/v/conv2d_5/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv2d_5/kernel/*
dtype0*
shape:@@*'
shared_nameAdam/v/conv2d_5/kernel

*Adam/v/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Й
Adam/m/conv2d_5/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv2d_5/kernel/*
dtype0*
shape:@@*'
shared_nameAdam/m/conv2d_5/kernel

*Adam/m/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
Ї
Adam/v/conv2d_4/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/conv2d_4/bias/*
dtype0*
shape:@*%
shared_nameAdam/v/conv2d_4/bias
y
(Adam/v/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/bias*
_output_shapes
:@*
dtype0
Ї
Adam/m/conv2d_4/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/conv2d_4/bias/*
dtype0*
shape:@*%
shared_nameAdam/m/conv2d_4/bias
y
(Adam/m/conv2d_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/bias*
_output_shapes
:@*
dtype0
Й
Adam/v/conv2d_4/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/conv2d_4/kernel/*
dtype0*
shape:@*'
shared_nameAdam/v/conv2d_4/kernel

*Adam/v/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_4/kernel*&
_output_shapes
:@*
dtype0
Й
Adam/m/conv2d_4/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/conv2d_4/kernel/*
dtype0*
shape:@*'
shared_nameAdam/m/conv2d_4/kernel

*Adam/m/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_4/kernel*&
_output_shapes
:@*
dtype0

learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0

	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:0*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:0*
dtype0

dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:@0*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@0*
dtype0

conv2d_5/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_5/bias/*
dtype0*
shape:@*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:@*
dtype0
Є
conv2d_5/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_5/kernel/*
dtype0*
shape:@@* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0

conv2d_4/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_4/bias/*
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
Є
conv2d_4/kernelVarHandleOp*
_output_shapes
: * 

debug_nameconv2d_4/kernel/*
dtype0*
shape:@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@*
dtype0

dense_5/biasVarHandleOp*
_output_shapes
: *

debug_namedense_5/bias/*
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

dense_5/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_5/kernel/*
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0

serving_default_input_7Placeholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ

serving_default_input_8Placeholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
л
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_8conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_243878

NoOpNoOp
пU
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*U
valueUBU BU
С
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
г
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
І
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
<
,0
-1
.2
/3
04
15
*6
+7*
<
,0
-1
.2
/3
04
15
*6
+7*
* 
А
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

7trace_0
8trace_1* 

9trace_0
:trace_1* 
* 

;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla*

Bserving_default* 
* 
Ш
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

,kernel
-bias
 I_jit_compiled_convolution_op*

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
Ѕ
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V_random_generator* 
Ш
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

.kernel
/bias
 ]_jit_compiled_convolution_op*

^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
Ѕ
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator* 

k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses* 
І
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

0kernel
1bias*
.
,0
-1
.2
/3
04
15*
.
,0
-1
.2
/3
04
15*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

|trace_0
}trace_1* 

~trace_0
trace_1* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 

*0
+1*

*0
+1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_4/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_4/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

0
1*
* 
* 
* 
* 
* 
* 

<0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
Ё16*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
0
1
2
3
4
5
6
 7*
D
0
1
2
3
4
5
6
Ё7*
r
Ђtrace_0
Ѓtrace_1
Єtrace_2
Ѕtrace_3
Іtrace_4
Їtrace_5
Јtrace_6
Љtrace_7* 
* 

,0
-1*

,0
-1*
* 

Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

Џtrace_0* 

Аtrace_0* 
* 
* 
* 
* 

Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 
* 
* 
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

Нtrace_0
Оtrace_1* 

Пtrace_0
Рtrace_1* 
* 

.0
/1*

.0
/1*
* 

Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Цtrace_0* 

Чtrace_0* 
* 
* 
* 
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

Эtrace_0* 

Юtrace_0* 
* 
* 
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

дtrace_0
еtrace_1* 

жtrace_0
зtrace_1* 
* 
* 
* 
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 

00
11*

00
11*
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

фtrace_0* 

хtrace_0* 
* 
C
0
1
2
3
4
5
6
7
8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ц	variables
ч	keras_api

шtotal

щcount*
M
ъ	variables
ы	keras_api

ьtotal

эcount
ю
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/conv2d_4/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_4/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_4/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_4/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_5/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_5/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/conv2d_5/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/conv2d_5/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_4/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_4/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_4/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_5/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ш0
щ1*

ц	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ь0
э1*

ъ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
к
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/bias	iterationlearning_rateAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biastotal_1count_1totalcountConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_244285
е
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_4/kerneldense_4/bias	iterationlearning_rateAdam/m/conv2d_4/kernelAdam/v/conv2d_4/kernelAdam/m/conv2d_4/biasAdam/v/conv2d_4/biasAdam/m/conv2d_5/kernelAdam/v/conv2d_5/kernelAdam/m/conv2d_5/biasAdam/v/conv2d_5/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biastotal_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_244384тй
Ќ
K
#__inference__update_step_xla_242751
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Щ

є
C__inference_dense_5_layer_call_and_return_conditional_losses_243736

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
б

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_243553

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Г
§
D__inference_conv2d_5_layer_call_and_return_conditional_losses_243535

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Р
Ї
"__inference__traced_restore_244384
file_prefix1
assignvariableop_dense_5_kernel:-
assignvariableop_1_dense_5_bias:<
"assignvariableop_2_conv2d_4_kernel:@.
 assignvariableop_3_conv2d_4_bias:@<
"assignvariableop_4_conv2d_5_kernel:@@.
 assignvariableop_5_conv2d_5_bias:@3
!assignvariableop_6_dense_4_kernel:@0-
assignvariableop_7_dense_4_bias:0&
assignvariableop_8_iteration:	 *
 assignvariableop_9_learning_rate: D
*assignvariableop_10_adam_m_conv2d_4_kernel:@D
*assignvariableop_11_adam_v_conv2d_4_kernel:@6
(assignvariableop_12_adam_m_conv2d_4_bias:@6
(assignvariableop_13_adam_v_conv2d_4_bias:@D
*assignvariableop_14_adam_m_conv2d_5_kernel:@@D
*assignvariableop_15_adam_v_conv2d_5_kernel:@@6
(assignvariableop_16_adam_m_conv2d_5_bias:@6
(assignvariableop_17_adam_v_conv2d_5_bias:@;
)assignvariableop_18_adam_m_dense_4_kernel:@0;
)assignvariableop_19_adam_v_dense_4_kernel:@05
'assignvariableop_20_adam_m_dense_4_bias:05
'assignvariableop_21_adam_v_dense_4_bias:0;
)assignvariableop_22_adam_m_dense_5_kernel:;
)assignvariableop_23_adam_v_dense_5_kernel:5
'assignvariableop_24_adam_m_dense_5_bias:5
'assignvariableop_25_adam_v_dense_5_bias:%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9с
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value§BњB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B К
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_8AssignVariableOpassignvariableop_8_iterationIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_9AssignVariableOp assignvariableop_9_learning_rateIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_10AssignVariableOp*assignvariableop_10_adam_m_conv2d_4_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_v_conv2d_4_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_12AssignVariableOp(assignvariableop_12_adam_m_conv2d_4_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_v_conv2d_4_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_m_conv2d_5_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_v_conv2d_5_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_conv2d_5_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_conv2d_5_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_m_dense_4_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_v_dense_4_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_m_dense_4_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_v_dense_4_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_5_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_5_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_5_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_5_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: Ќ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_31Identity_31:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_4/bias:)	%
#
_user_specified_name	iteration:-
)
'
_user_specified_namelearning_rate:62
0
_user_specified_nameAdam/m/conv2d_4/kernel:62
0
_user_specified_nameAdam/v/conv2d_4/kernel:40
.
_user_specified_nameAdam/m/conv2d_4/bias:40
.
_user_specified_nameAdam/v/conv2d_4/bias:62
0
_user_specified_nameAdam/m/conv2d_5/kernel:62
0
_user_specified_nameAdam/v/conv2d_5/kernel:40
.
_user_specified_nameAdam/m/conv2d_5/bias:40
.
_user_specified_nameAdam/v/conv2d_5/bias:51
/
_user_specified_nameAdam/m/dense_4/kernel:51
/
_user_specified_nameAdam/v/dense_4/kernel:3/
-
_user_specified_nameAdam/m/dense_4/bias:3/
-
_user_specified_nameAdam/v/dense_4/bias:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount

g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_244025

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і	
є
C__inference_dense_4_layer_call_and_return_conditional_losses_244082

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
а
W
#__inference__update_step_xla_242726
gradient"
variable:@@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@@: *
	_noinline(:P L
&
_output_shapes
:@@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
И
O
#__inference__update_step_xla_242736
gradient
variable:@0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@0: *
	_noinline(:H D

_output_shapes

:@0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable

е
(__inference_model_5_layer_call_fn_243832
input_7
input_8!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_243788o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:&"
 
_user_specified_name243814:&"
 
_user_specified_name243816:&"
 
_user_specified_name243818:&"
 
_user_specified_name243820:&"
 
_user_specified_name243822:&"
 
_user_specified_name243824:&"
 
_user_specified_name243826:&	"
 
_user_specified_name243828


)__inference_conv2d_4_layer_call_fn_243947

inputs!
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243505w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:&"
 
_user_specified_name243941:&"
 
_user_specified_name243943
ф
б
$__inference_signature_wrapper_243878
input_7
input_8!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_243461o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:&"
 
_user_specified_name243860:&"
 
_user_specified_name243862:&"
 
_user_specified_name243864:&"
 
_user_specified_name243866:&"
 
_user_specified_name243868:&"
 
_user_specified_name243870:&"
 
_user_specified_name243872:&	"
 
_user_specified_name243874
И
O
#__inference__update_step_xla_242746
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
а
W
#__inference__update_step_xla_242716
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:P L
&
_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
б

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_243990

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
б

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_244047

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Г%
ч
C__inference_model_4_layer_call_and_return_conditional_losses_243572
input_9)
conv2d_4_243506:@
conv2d_4_243508:@)
conv2d_5_243536:@@
conv2d_5_243538:@ 
dense_4_243566:@0
dense_4_243568:0
identityЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ!dropout_4/StatefulPartitionedCallЂ!dropout_5/StatefulPartitionedCallќ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_4_243506conv2d_4_243508*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243505є
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243466ї
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_243523
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0conv2d_5_243536conv2d_5_243538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_243535є
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_243476
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_243553
*global_average_pooling2d_2/PartitionedCallPartitionedCall*dropout_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_243487
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0dense_4_243566dense_4_243568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_243565w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0в
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_9:&"
 
_user_specified_name243506:&"
 
_user_specified_name243508:&"
 
_user_specified_name243536:&"
 
_user_specified_name243538:&"
 
_user_specified_name243566:&"
 
_user_specified_name243568
Ќ
K
#__inference__update_step_xla_242721
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Т
F
*__inference_dropout_4_layer_call_fn_243978

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_243585h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Щ

є
C__inference_dense_5_layer_call_and_return_conditional_losses_243938

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
і	
є
C__inference_dense_4_layer_call_and_return_conditional_losses_243565

inputs0
matmul_readvariableop_resource:@0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource


n
D__inference_lambda_2_layer_call_and_return_conditional_losses_243780

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
З
r
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_244063

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_244052

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ы
ш
C__inference_model_5_layer_call_and_return_conditional_losses_243743
input_7
input_8(
model_4_243691:@
model_4_243693:@(
model_4_243695:@@
model_4_243697:@ 
model_4_243699:@0
model_4_243701:0 
dense_5_243737:
dense_5_243739:
identityЂdense_5/StatefulPartitionedCallЂmodel_4/StatefulPartitionedCallЂ!model_4/StatefulPartitionedCall_1И
model_4/StatefulPartitionedCallStatefulPartitionedCallinput_7model_4_243691model_4_243693model_4_243695model_4_243697model_4_243699model_4_243701*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243572К
!model_4/StatefulPartitionedCall_1StatefulPartitionedCallinput_8model_4_243691model_4_243693model_4_243695model_4_243697model_4_243699model_4_243701*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243572
lambda_2/PartitionedCallPartitionedCall(model_4/StatefulPartitionedCall:output:0*model_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_243724
dense_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0dense_5_243737dense_5_243739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_243736w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_5/StatefulPartitionedCall ^model_4/StatefulPartitionedCall"^model_4/StatefulPartitionedCall_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2F
!model_4/StatefulPartitionedCall_1!model_4/StatefulPartitionedCall_1:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:&"
 
_user_specified_name243691:&"
 
_user_specified_name243693:&"
 
_user_specified_name243695:&"
 
_user_specified_name243697:&"
 
_user_specified_name243699:&"
 
_user_specified_name243701:&"
 
_user_specified_name243737:&	"
 
_user_specified_name243739
ы
ш
C__inference_model_5_layer_call_and_return_conditional_losses_243788
input_7
input_8(
model_4_243747:@
model_4_243749:@(
model_4_243751:@@
model_4_243753:@ 
model_4_243755:@0
model_4_243757:0 
dense_5_243782:
dense_5_243784:
identityЂdense_5/StatefulPartitionedCallЂmodel_4/StatefulPartitionedCallЂ!model_4/StatefulPartitionedCall_1И
model_4/StatefulPartitionedCallStatefulPartitionedCallinput_7model_4_243747model_4_243749model_4_243751model_4_243753model_4_243755model_4_243757*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243606К
!model_4/StatefulPartitionedCall_1StatefulPartitionedCallinput_8model_4_243747model_4_243749model_4_243751model_4_243753model_4_243755model_4_243757*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243606
lambda_2/PartitionedCallPartitionedCall(model_4/StatefulPartitionedCall:output:0*model_4/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_243780
dense_5/StatefulPartitionedCallStatefulPartitionedCall!lambda_2/PartitionedCall:output:0dense_5_243782dense_5_243784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_243736w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_5/StatefulPartitionedCall ^model_4/StatefulPartitionedCall"^model_4/StatefulPartitionedCall_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
model_4/StatefulPartitionedCallmodel_4/StatefulPartitionedCall2F
!model_4/StatefulPartitionedCall_1!model_4/StatefulPartitionedCall_1:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:&"
 
_user_specified_name243747:&"
 
_user_specified_name243749:&"
 
_user_specified_name243751:&"
 
_user_specified_name243753:&"
 
_user_specified_name243755:&"
 
_user_specified_name243757:&"
 
_user_specified_name243782:&	"
 
_user_specified_name243784
я

(__inference_dense_4_layer_call_fn_244072

inputs
unknown:@0
	unknown_0:0
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_243565o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:&"
 
_user_specified_name244066:&"
 
_user_specified_name244068
Г
§
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243505

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
№
c
*__inference_dropout_4_layer_call_fn_243973

inputs
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_243523w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

W
;__inference_global_average_pooling2d_2_layer_call_fn_244057

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_243487i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_243597

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

е
(__inference_model_5_layer_call_fn_243810
input_7
input_8!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_243743o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:&"
 
_user_specified_name243792:&"
 
_user_specified_name243794:&"
 
_user_specified_name243796:&"
 
_user_specified_name243798:&"
 
_user_specified_name243800:&"
 
_user_specified_name243802:&"
 
_user_specified_name243804:&	"
 
_user_specified_name243806

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243466

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П"

C__inference_model_4_layer_call_and_return_conditional_losses_243606
input_9)
conv2d_4_243575:@
conv2d_4_243577:@)
conv2d_5_243587:@@
conv2d_5_243589:@ 
dense_4_243600:@0
dense_4_243602:0
identityЂ conv2d_4/StatefulPartitionedCallЂ conv2d_5/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallќ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_4_243575conv2d_4_243577*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243505є
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243466ч
dropout_4/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_243585
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0conv2d_5_243587conv2d_5_243589*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_243535є
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_243476ч
dropout_5/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_243597ћ
*global_average_pooling2d_2/PartitionedCallPartitionedCall"dropout_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *_
fZRX
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_243487
dense_4/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling2d_2/PartitionedCall:output:0dense_4_243600dense_4_243602*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_243565w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0
NoOpNoOp!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_9:&"
 
_user_specified_name243575:&"
 
_user_specified_name243577:&"
 
_user_specified_name243587:&"
 
_user_specified_name243589:&"
 
_user_specified_name243600:&"
 
_user_specified_name243602
Т
F
*__inference_dropout_5_layer_call_fn_244035

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_243597h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_243476

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
§
D__inference_conv2d_5_layer_call_and_return_conditional_losses_244015

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ј
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_243585

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243968

inputs
identityЂ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_242731
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
б

d
E__inference_dropout_4_layer_call_and_return_conditional_losses_243523

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ў
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


)__inference_conv2d_5_layer_call_fn_244004

inputs!
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_243535w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:&"
 
_user_specified_name243998:&"
 
_user_specified_name244000
З
r
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_243487

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ѕ
U
)__inference_lambda_2_layer_call_fn_243884
inputs_0
inputs_1
identityП
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_243724`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:Q M
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_1


p
D__inference_lambda_2_layer_call_and_return_conditional_losses_243904
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:Q M
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_1
к


(__inference_model_4_layer_call_fn_243623
input_9!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_9:&"
 
_user_specified_name243609:&"
 
_user_specified_name243611:&"
 
_user_specified_name243613:&"
 
_user_specified_name243615:&"
 
_user_specified_name243617:&"
 
_user_specified_name243619
Л
L
0__inference_max_pooling2d_4_layer_call_fn_243963

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243466
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ќ
K
#__inference__update_step_xla_242741
gradient
variable:0*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:0: *
	_noinline(:D @

_output_shapes
:0
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
шi
С

!__inference__wrapped_model_243461
input_7
input_8Q
7model_5_model_4_conv2d_4_conv2d_readvariableop_resource:@F
8model_5_model_4_conv2d_4_biasadd_readvariableop_resource:@Q
7model_5_model_4_conv2d_5_conv2d_readvariableop_resource:@@F
8model_5_model_4_conv2d_5_biasadd_readvariableop_resource:@H
6model_5_model_4_dense_4_matmul_readvariableop_resource:@0E
7model_5_model_4_dense_4_biasadd_readvariableop_resource:0@
.model_5_dense_5_matmul_readvariableop_resource:=
/model_5_dense_5_biasadd_readvariableop_resource:
identityЂ&model_5/dense_5/BiasAdd/ReadVariableOpЂ%model_5/dense_5/MatMul/ReadVariableOpЂ/model_5/model_4/conv2d_4/BiasAdd/ReadVariableOpЂ1model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOpЂ.model_5/model_4/conv2d_4/Conv2D/ReadVariableOpЂ0model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOpЂ/model_5/model_4/conv2d_5/BiasAdd/ReadVariableOpЂ1model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOpЂ.model_5/model_4/conv2d_5/Conv2D/ReadVariableOpЂ0model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOpЂ.model_5/model_4/dense_4/BiasAdd/ReadVariableOpЂ0model_5/model_4/dense_4/BiasAdd_1/ReadVariableOpЂ-model_5/model_4/dense_4/MatMul/ReadVariableOpЂ/model_5/model_4/dense_4/MatMul_1/ReadVariableOpЎ
.model_5/model_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp7model_5_model_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ь
model_5/model_4/conv2d_4/Conv2DConv2Dinput_76model_5/model_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Є
/model_5/model_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp8model_5_model_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
 model_5/model_4/conv2d_4/BiasAddBiasAdd(model_5/model_4/conv2d_4/Conv2D:output:07model_5/model_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_5/model_4/conv2d_4/ReluRelu)model_5/model_4/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ь
'model_5/model_4/max_pooling2d_4/MaxPoolMaxPool+model_5/model_4/conv2d_4/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

"model_5/model_4/dropout_4/IdentityIdentity0model_5/model_4/max_pooling2d_4/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ў
.model_5/model_4/conv2d_5/Conv2D/ReadVariableOpReadVariableOp7model_5_model_4_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0№
model_5/model_4/conv2d_5/Conv2DConv2D+model_5/model_4/dropout_4/Identity:output:06model_5/model_4/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
Є
/model_5/model_4/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp8model_5_model_4_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
 model_5/model_4/conv2d_5/BiasAddBiasAdd(model_5/model_4/conv2d_5/Conv2D:output:07model_5/model_4/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_5/model_4/conv2d_5/ReluRelu)model_5/model_4/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ь
'model_5/model_4/max_pooling2d_5/MaxPoolMaxPool+model_5/model_4/conv2d_5/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

"model_5/model_4/dropout_5/IdentityIdentity0model_5/model_4/max_pooling2d_5/MaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
Amodel_5/model_4/global_average_pooling2d_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      т
/model_5/model_4/global_average_pooling2d_2/MeanMean+model_5/model_4/dropout_5/Identity:output:0Jmodel_5/model_4/global_average_pooling2d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
-model_5/model_4/dense_4/MatMul/ReadVariableOpReadVariableOp6model_5_model_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0Ы
model_5/model_4/dense_4/MatMulMatMul8model_5/model_4/global_average_pooling2d_2/Mean:output:05model_5/model_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0Ђ
.model_5/model_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp7model_5_model_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0О
model_5/model_4/dense_4/BiasAddBiasAdd(model_5/model_4/dense_4/MatMul:product:06model_5/model_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0А
0model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOpReadVariableOp7model_5_model_4_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0а
!model_5/model_4/conv2d_4/Conv2D_1Conv2Dinput_88model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
І
1model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOpReadVariableOp8model_5_model_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
"model_5/model_4/conv2d_4/BiasAdd_1BiasAdd*model_5/model_4/conv2d_4/Conv2D_1:output:09model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_5/model_4/conv2d_4/Relu_1Relu+model_5/model_4/conv2d_4/BiasAdd_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@а
)model_5/model_4/max_pooling2d_4/MaxPool_1MaxPool-model_5/model_4/conv2d_4/Relu_1:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

$model_5/model_4/dropout_4/Identity_1Identity2model_5/model_4/max_pooling2d_4/MaxPool_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@А
0model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOpReadVariableOp7model_5_model_4_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0і
!model_5/model_4/conv2d_5/Conv2D_1Conv2D-model_5/model_4/dropout_4/Identity_1:output:08model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
І
1model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOpReadVariableOp8model_5_model_4_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ю
"model_5/model_4/conv2d_5/BiasAdd_1BiasAdd*model_5/model_4/conv2d_5/Conv2D_1:output:09model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
model_5/model_4/conv2d_5/Relu_1Relu+model_5/model_4/conv2d_5/BiasAdd_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@а
)model_5/model_4/max_pooling2d_5/MaxPool_1MaxPool-model_5/model_4/conv2d_5/Relu_1:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingVALID*
strides

$model_5/model_4/dropout_5/Identity_1Identity2model_5/model_4/max_pooling2d_5/MaxPool_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
Cmodel_5/model_4/global_average_pooling2d_2/Mean_1/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ш
1model_5/model_4/global_average_pooling2d_2/Mean_1Mean-model_5/model_4/dropout_5/Identity_1:output:0Lmodel_5/model_4/global_average_pooling2d_2/Mean_1/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@І
/model_5/model_4/dense_4/MatMul_1/ReadVariableOpReadVariableOp6model_5_model_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype0б
 model_5/model_4/dense_4/MatMul_1MatMul:model_5/model_4/global_average_pooling2d_2/Mean_1:output:07model_5/model_4/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0Є
0model_5/model_4/dense_4/BiasAdd_1/ReadVariableOpReadVariableOp7model_5_model_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ф
!model_5/model_4/dense_4/BiasAdd_1BiasAdd*model_5/model_4/dense_4/MatMul_1:product:08model_5/model_4/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ0Ѓ
model_5/lambda_2/subSub(model_5/model_4/dense_4/BiasAdd:output:0*model_5/model_4/dense_4/BiasAdd_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ0m
model_5/lambda_2/SquareSquaremodel_5/lambda_2/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ0h
&model_5/lambda_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ќ
model_5/lambda_2/SumSummodel_5/lambda_2/Square:y:0/model_5/lambda_2/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(_
model_5/lambda_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
model_5/lambda_2/MaximumMaximummodel_5/lambda_2/Sum:output:0#model_5/lambda_2/Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ[
model_5/lambda_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
model_5/lambda_2/Maximum_1Maximummodel_5/lambda_2/Maximum:z:0model_5/lambda_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџo
model_5/lambda_2/SqrtSqrtmodel_5/lambda_2/Maximum_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model_5/dense_5/MatMul/ReadVariableOpReadVariableOp.model_5_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
model_5/dense_5/MatMulMatMulmodel_5/lambda_2/Sqrt:y:0-model_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_5/dense_5/BiasAddBiasAdd model_5/dense_5/MatMul:product:0.model_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
model_5/dense_5/SigmoidSigmoid model_5/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitymodel_5/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЭ
NoOpNoOp'^model_5/dense_5/BiasAdd/ReadVariableOp&^model_5/dense_5/MatMul/ReadVariableOp0^model_5/model_4/conv2d_4/BiasAdd/ReadVariableOp2^model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOp/^model_5/model_4/conv2d_4/Conv2D/ReadVariableOp1^model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOp0^model_5/model_4/conv2d_5/BiasAdd/ReadVariableOp2^model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOp/^model_5/model_4/conv2d_5/Conv2D/ReadVariableOp1^model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOp/^model_5/model_4/dense_4/BiasAdd/ReadVariableOp1^model_5/model_4/dense_4/BiasAdd_1/ReadVariableOp.^model_5/model_4/dense_4/MatMul/ReadVariableOp0^model_5/model_4/dense_4/MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : 2P
&model_5/dense_5/BiasAdd/ReadVariableOp&model_5/dense_5/BiasAdd/ReadVariableOp2N
%model_5/dense_5/MatMul/ReadVariableOp%model_5/dense_5/MatMul/ReadVariableOp2b
/model_5/model_4/conv2d_4/BiasAdd/ReadVariableOp/model_5/model_4/conv2d_4/BiasAdd/ReadVariableOp2f
1model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOp1model_5/model_4/conv2d_4/BiasAdd_1/ReadVariableOp2`
.model_5/model_4/conv2d_4/Conv2D/ReadVariableOp.model_5/model_4/conv2d_4/Conv2D/ReadVariableOp2d
0model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOp0model_5/model_4/conv2d_4/Conv2D_1/ReadVariableOp2b
/model_5/model_4/conv2d_5/BiasAdd/ReadVariableOp/model_5/model_4/conv2d_5/BiasAdd/ReadVariableOp2f
1model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOp1model_5/model_4/conv2d_5/BiasAdd_1/ReadVariableOp2`
.model_5/model_4/conv2d_5/Conv2D/ReadVariableOp.model_5/model_4/conv2d_5/Conv2D/ReadVariableOp2d
0model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOp0model_5/model_4/conv2d_5/Conv2D_1/ReadVariableOp2`
.model_5/model_4/dense_4/BiasAdd/ReadVariableOp.model_5/model_4/dense_4/BiasAdd/ReadVariableOp2d
0model_5/model_4/dense_4/BiasAdd_1/ReadVariableOp0model_5/model_4/dense_4/BiasAdd_1/ReadVariableOp2^
-model_5/model_4/dense_4/MatMul/ReadVariableOp-model_5/model_4/dense_4/MatMul/ReadVariableOp2b
/model_5/model_4/dense_4/MatMul_1/ReadVariableOp/model_5/model_4/dense_4/MatMul_1/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource


p
D__inference_lambda_2_layer_call_and_return_conditional_losses_243918
inputs_0
inputs_1
identityP
subSubinputs_0inputs_1*
T0*'
_output_shapes
:џџџџџџџџџ0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:Q M
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_1
Ѕ
U
)__inference_lambda_2_layer_call_fn_243890
inputs_0
inputs_1
identityП
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lambda_2_layer_call_and_return_conditional_losses_243780`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:Q M
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ0
"
_user_specified_name
inputs_1
Г
§
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243958

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ј
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_243995

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:џџџџџџџџџ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Л
L
0__inference_max_pooling2d_5_layer_call_fn_244020

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_243476
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч

__inference__traced_save_244285
file_prefix7
%read_disablecopyonread_dense_5_kernel:3
%read_1_disablecopyonread_dense_5_bias:B
(read_2_disablecopyonread_conv2d_4_kernel:@4
&read_3_disablecopyonread_conv2d_4_bias:@B
(read_4_disablecopyonread_conv2d_5_kernel:@@4
&read_5_disablecopyonread_conv2d_5_bias:@9
'read_6_disablecopyonread_dense_4_kernel:@03
%read_7_disablecopyonread_dense_4_bias:0,
"read_8_disablecopyonread_iteration:	 0
&read_9_disablecopyonread_learning_rate: J
0read_10_disablecopyonread_adam_m_conv2d_4_kernel:@J
0read_11_disablecopyonread_adam_v_conv2d_4_kernel:@<
.read_12_disablecopyonread_adam_m_conv2d_4_bias:@<
.read_13_disablecopyonread_adam_v_conv2d_4_bias:@J
0read_14_disablecopyonread_adam_m_conv2d_5_kernel:@@J
0read_15_disablecopyonread_adam_v_conv2d_5_kernel:@@<
.read_16_disablecopyonread_adam_m_conv2d_5_bias:@<
.read_17_disablecopyonread_adam_v_conv2d_5_bias:@A
/read_18_disablecopyonread_adam_m_dense_4_kernel:@0A
/read_19_disablecopyonread_adam_v_dense_4_kernel:@0;
-read_20_disablecopyonread_adam_m_dense_4_bias:0;
-read_21_disablecopyonread_adam_v_dense_4_bias:0A
/read_22_disablecopyonread_adam_m_dense_5_kernel:A
/read_23_disablecopyonread_adam_v_dense_5_kernel:;
-read_24_disablecopyonread_adam_m_dense_5_bias:;
-read_25_disablecopyonread_adam_v_dense_5_bias:+
!read_26_disablecopyonread_total_1: +
!read_27_disablecopyonread_count_1: )
read_28_disablecopyonread_total: )
read_29_disablecopyonread_count: 
savev2_const
identity_61ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_5_kernel"/device:CPU:0*
_output_shapes
 Ё
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_5_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_5_bias"/device:CPU:0*
_output_shapes
 Ё
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_5_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 А
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_conv2d_4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0u

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:@z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_conv2d_4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 А
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_conv2d_5_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_conv2d_5_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_4_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@0*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@0e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@0y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 Ё
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_4_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:0v
Read_8/DisableCopyOnReadDisableCopyOnRead"read_8_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_8/ReadVariableOpReadVariableOp"read_8_disablecopyonread_iteration^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_9/DisableCopyOnReadDisableCopyOnRead&read_9_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 
Read_9/ReadVariableOpReadVariableOp&read_9_disablecopyonread_learning_rate^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_10/DisableCopyOnReadDisableCopyOnRead0read_10_disablecopyonread_adam_m_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 К
Read_10/ReadVariableOpReadVariableOp0read_10_disablecopyonread_adam_m_conv2d_4_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:@
Read_11/DisableCopyOnReadDisableCopyOnRead0read_11_disablecopyonread_adam_v_conv2d_4_kernel"/device:CPU:0*
_output_shapes
 К
Read_11/ReadVariableOpReadVariableOp0read_11_disablecopyonread_adam_v_conv2d_4_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
:@
Read_12/DisableCopyOnReadDisableCopyOnRead.read_12_disablecopyonread_adam_m_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_12/ReadVariableOpReadVariableOp.read_12_disablecopyonread_adam_m_conv2d_4_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_13/DisableCopyOnReadDisableCopyOnRead.read_13_disablecopyonread_adam_v_conv2d_4_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_13/ReadVariableOpReadVariableOp.read_13_disablecopyonread_adam_v_conv2d_4_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_14/DisableCopyOnReadDisableCopyOnRead0read_14_disablecopyonread_adam_m_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 К
Read_14/ReadVariableOpReadVariableOp0read_14_disablecopyonread_adam_m_conv2d_5_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@
Read_15/DisableCopyOnReadDisableCopyOnRead0read_15_disablecopyonread_adam_v_conv2d_5_kernel"/device:CPU:0*
_output_shapes
 К
Read_15/ReadVariableOpReadVariableOp0read_15_disablecopyonread_adam_v_conv2d_5_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@
Read_16/DisableCopyOnReadDisableCopyOnRead.read_16_disablecopyonread_adam_m_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_16/ReadVariableOpReadVariableOp.read_16_disablecopyonread_adam_m_conv2d_5_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_adam_v_conv2d_5_bias"/device:CPU:0*
_output_shapes
 Ќ
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_adam_v_conv2d_5_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_m_dense_4_kernel"/device:CPU:0*
_output_shapes
 Б
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_m_dense_4_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@0*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@0e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:@0
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_v_dense_4_kernel"/device:CPU:0*
_output_shapes
 Б
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_v_dense_4_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@0*
dtype0o
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@0e
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes

:@0
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_m_dense_4_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_m_dense_4_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:0
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_v_dense_4_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_v_dense_4_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:0*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:0a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:0
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_dense_5_kernel"/device:CPU:0*
_output_shapes
 Б
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_dense_5_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_dense_5_kernel"/device:CPU:0*
_output_shapes
 Б
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_dense_5_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_m_dense_5_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_m_dense_5_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_adam_v_dense_5_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_adam_v_dense_5_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_26/DisableCopyOnReadDisableCopyOnRead!read_26_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_26/ReadVariableOpReadVariableOp!read_26_disablecopyonread_total_1^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_27/DisableCopyOnReadDisableCopyOnRead!read_27_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_27/ReadVariableOpReadVariableOp!read_27_disablecopyonread_count_1^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_28/DisableCopyOnReadDisableCopyOnReadread_28_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_28/ReadVariableOpReadVariableOpread_28_disablecopyonread_total^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_29/DisableCopyOnReadDisableCopyOnReadread_29_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_29/ReadVariableOpReadVariableOpread_29_disablecopyonread_count^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: о
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value§BњB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_60Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_61IdentityIdentity_60:output:0^NoOp*
T0*
_output_shapes
: е
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:.*
(
_user_specified_namedense_5/kernel:,(
&
_user_specified_namedense_5/bias:/+
)
_user_specified_nameconv2d_4/kernel:-)
'
_user_specified_nameconv2d_4/bias:/+
)
_user_specified_nameconv2d_5/kernel:-)
'
_user_specified_nameconv2d_5/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_4/bias:)	%
#
_user_specified_name	iteration:-
)
'
_user_specified_namelearning_rate:62
0
_user_specified_nameAdam/m/conv2d_4/kernel:62
0
_user_specified_nameAdam/v/conv2d_4/kernel:40
.
_user_specified_nameAdam/m/conv2d_4/bias:40
.
_user_specified_nameAdam/v/conv2d_4/bias:62
0
_user_specified_nameAdam/m/conv2d_5/kernel:62
0
_user_specified_nameAdam/v/conv2d_5/kernel:40
.
_user_specified_nameAdam/m/conv2d_5/bias:40
.
_user_specified_nameAdam/v/conv2d_5/bias:51
/
_user_specified_nameAdam/m/dense_4/kernel:51
/
_user_specified_nameAdam/v/dense_4/kernel:3/
-
_user_specified_nameAdam/m/dense_4/bias:3/
-
_user_specified_nameAdam/v/dense_4/bias:51
/
_user_specified_nameAdam/m/dense_5/kernel:51
/
_user_specified_nameAdam/v/dense_5/kernel:3/
-
_user_specified_nameAdam/m/dense_5/bias:3/
-
_user_specified_nameAdam/v/dense_5/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:=9

_output_shapes
: 

_user_specified_nameConst
к


(__inference_model_4_layer_call_fn_243640
input_9!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@0
	unknown_4:0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_4_layer_call_and_return_conditional_losses_243606o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ0<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_9:&"
 
_user_specified_name243626:&"
 
_user_specified_name243628:&"
 
_user_specified_name243630:&"
 
_user_specified_name243632:&"
 
_user_specified_name243634:&"
 
_user_specified_name243636
я

(__inference_dense_5_layer_call_fn_243927

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_243736o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:&"
 
_user_specified_name243921:&"
 
_user_specified_name243923
№
c
*__inference_dropout_5_layer_call_fn_244030

inputs
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_243553w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


n
D__inference_lambda_2_layer_call_and_return_conditional_losses_243724

inputs
inputs_1
identityN
subSubinputsinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ0K
SquareSquaresub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ0W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
SumSum
Square:y:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
	keep_dims(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3f
MaximumMaximumSum:output:0Maximum/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    c
	Maximum_1MaximumMaximum:z:0Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
SqrtSqrtMaximum_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџP
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ0:џџџџџџџџџ0:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ї
serving_defaultу
C
input_78
serving_default_input_7:0џџџџџџџџџ
C
input_88
serving_default_input_8:0џџџџџџџџџ;
dense_50
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЌА
и
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
ъ
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
Ѕ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
X
,0
-1
.2
/3
04
15
*6
+7"
trackable_list_wrapper
X
,0
-1
.2
/3
04
15
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
У
7trace_0
8trace_12
(__inference_model_5_layer_call_fn_243810
(__inference_model_5_layer_call_fn_243832Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z7trace_0z8trace_1
љ
9trace_0
:trace_12Т
C__inference_model_5_layer_call_and_return_conditional_losses_243743
C__inference_model_5_layer_call_and_return_conditional_losses_243788Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z9trace_0z:trace_1
еBв
!__inference__wrapped_model_243461input_7input_8"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla"
experimentalOptimizer
,
Bserving_default"
signature_map
"
_tf_keras_input_layer
н
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

,kernel
-bias
 I_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
М
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V_random_generator"
_tf_keras_layer
н
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

.kernel
/bias
 ]_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
М
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses
j_random_generator"
_tf_keras_layer
Ѕ
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

0kernel
1bias"
_tf_keras_layer
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
У
|trace_0
}trace_12
(__inference_model_4_layer_call_fn_243623
(__inference_model_4_layer_call_fn_243640Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z|trace_0z}trace_1
љ
~trace_0
trace_12Т
C__inference_model_4_layer_call_and_return_conditional_losses_243572
C__inference_model_4_layer_call_and_return_conditional_losses_243606Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0ztrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Щ
trace_0
trace_12
)__inference_lambda_2_layer_call_fn_243884
)__inference_lambda_2_layer_call_fn_243890Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
џ
trace_0
trace_12Ф
D__inference_lambda_2_layer_call_and_return_conditional_losses_243904
D__inference_lambda_2_layer_call_and_return_conditional_losses_243918Е
ЎВЊ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЂ

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ф
trace_02Х
(__inference_dense_5_layer_call_fn_243927
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
џ
trace_02р
C__inference_dense_5_layer_call_and_return_conditional_losses_243938
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :2dense_5/kernel
:2dense_5/bias
):'@2conv2d_4/kernel
:@2conv2d_4/bias
):'@@2conv2d_5/kernel
:@2conv2d_5/bias
 :@02dense_4/kernel
:02dense_4/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bэ
(__inference_model_5_layer_call_fn_243810input_7input_8"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№Bэ
(__inference_model_5_layer_call_fn_243832input_7input_8"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_model_5_layer_call_and_return_conditional_losses_243743input_7input_8"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_model_5_layer_call_and_return_conditional_losses_243788input_7input_8"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ў
<0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
Ё16"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
`
0
1
2
3
4
5
6
Ё7"
trackable_list_wrapper
Н
Ђtrace_0
Ѓtrace_1
Єtrace_2
Ѕtrace_3
Іtrace_4
Їtrace_5
Јtrace_6
Љtrace_72к
#__inference__update_step_xla_242716
#__inference__update_step_xla_242721
#__inference__update_step_xla_242726
#__inference__update_step_xla_242731
#__inference__update_step_xla_242736
#__inference__update_step_xla_242741
#__inference__update_step_xla_242746
#__inference__update_step_xla_242751Џ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zЂtrace_0zЃtrace_1zЄtrace_2zЅtrace_3zІtrace_4zЇtrace_5zЈtrace_6zЉtrace_7
тBп
$__inference_signature_wrapper_243878input_7input_8"Є
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 '

kwonlyargs
	jinput_7
	jinput_8
kwonlydefaults
 
annotationsЊ *
 
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Њnon_trainable_variables
Ћlayers
Ќmetrics
 ­layer_regularization_losses
Ўlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
х
Џtrace_02Ц
)__inference_conv2d_4_layer_call_fn_243947
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЏtrace_0

Аtrace_02с
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243958
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ь
Жtrace_02Э
0__inference_max_pooling2d_4_layer_call_fn_243963
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0

Зtrace_02ш
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243968
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
П
Нtrace_0
Оtrace_12
*__inference_dropout_4_layer_call_fn_243973
*__inference_dropout_4_layer_call_fn_243978Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zНtrace_0zОtrace_1
ѕ
Пtrace_0
Рtrace_12К
E__inference_dropout_4_layer_call_and_return_conditional_losses_243990
E__inference_dropout_4_layer_call_and_return_conditional_losses_243995Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0zРtrace_1
"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
х
Цtrace_02Ц
)__inference_conv2d_5_layer_call_fn_244004
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0

Чtrace_02с
D__inference_conv2d_5_layer_call_and_return_conditional_losses_244015
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЧtrace_0
Њ2ЇЄ
В
FullArgSpec
args
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ь
Эtrace_02Э
0__inference_max_pooling2d_5_layer_call_fn_244020
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0

Юtrace_02ш
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_244025
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
П
дtrace_0
еtrace_12
*__inference_dropout_5_layer_call_fn_244030
*__inference_dropout_5_layer_call_fn_244035Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0zеtrace_1
ѕ
жtrace_0
зtrace_12К
E__inference_dropout_5_layer_call_and_return_conditional_losses_244047
E__inference_dropout_5_layer_call_and_return_conditional_losses_244052Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zжtrace_0zзtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
ї
нtrace_02и
;__inference_global_average_pooling2d_2_layer_call_fn_244057
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0

оtrace_02ѓ
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_244063
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zоtrace_0
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
В
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
ф
фtrace_02Х
(__inference_dense_4_layer_call_fn_244072
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zфtrace_0
џ
хtrace_02р
C__inference_dense_4_layer_call_and_return_conditional_losses_244082
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zхtrace_0
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
(__inference_model_4_layer_call_fn_243623input_9"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
чBф
(__inference_model_4_layer_call_fn_243640input_9"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
C__inference_model_4_layer_call_and_return_conditional_losses_243572input_9"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bџ
C__inference_model_4_layer_call_and_return_conditional_losses_243606input_9"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB№
)__inference_lambda_2_layer_call_fn_243884inputs_0inputs_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
)__inference_lambda_2_layer_call_fn_243890inputs_0inputs_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lambda_2_layer_call_and_return_conditional_losses_243904inputs_0inputs_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
D__inference_lambda_2_layer_call_and_return_conditional_losses_243918inputs_0inputs_1"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_dense_5_layer_call_fn_243927inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_5_layer_call_and_return_conditional_losses_243938inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
ц	variables
ч	keras_api

шtotal

щcount"
_tf_keras_metric
c
ъ	variables
ы	keras_api

ьtotal

эcount
ю
_fn_kwargs"
_tf_keras_metric
.:,@2Adam/m/conv2d_4/kernel
.:,@2Adam/v/conv2d_4/kernel
 :@2Adam/m/conv2d_4/bias
 :@2Adam/v/conv2d_4/bias
.:,@@2Adam/m/conv2d_5/kernel
.:,@@2Adam/v/conv2d_5/kernel
 :@2Adam/m/conv2d_5/bias
 :@2Adam/v/conv2d_5/bias
%:#@02Adam/m/dense_4/kernel
%:#@02Adam/v/dense_4/kernel
:02Adam/m/dense_4/bias
:02Adam/v/dense_4/bias
%:#2Adam/m/dense_5/kernel
%:#2Adam/v/dense_5/kernel
:2Adam/m/dense_5/bias
:2Adam/v/dense_5/bias
юBы
#__inference__update_step_xla_242716gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242721gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242726gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242731gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242736gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242741gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242746gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
#__inference__update_step_xla_242751gradientvariable"­
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv2d_4_layer_call_fn_243947inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243958inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
0__inference_max_pooling2d_4_layer_call_fn_243963inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243968inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
*__inference_dropout_4_layer_call_fn_243973inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
рBн
*__inference_dropout_4_layer_call_fn_243978inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_dropout_4_layer_call_and_return_conditional_losses_243990inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_dropout_4_layer_call_and_return_conditional_losses_243995inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
гBа
)__inference_conv2d_5_layer_call_fn_244004inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
D__inference_conv2d_5_layer_call_and_return_conditional_losses_244015inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
0__inference_max_pooling2d_5_layer_call_fn_244020inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕBђ
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_244025inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
*__inference_dropout_5_layer_call_fn_244030inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
рBн
*__inference_dropout_5_layer_call_fn_244035inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_dropout_5_layer_call_and_return_conditional_losses_244047inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
E__inference_dropout_5_layer_call_and_return_conditional_losses_244052inputs"Є
В
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
;__inference_global_average_pooling2d_2_layer_call_fn_244057inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_244063inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_dense_4_layer_call_fn_244072inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_4_layer_call_and_return_conditional_losses_244082inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
ш0
щ1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
:  (2total
:  (2count
0
ь0
э1"
trackable_list_wrapper
.
ъ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЅ
#__inference__update_step_xla_242716~xЂu
nЂk
!
gradient@
<9	%Ђ"
њ@

p
` VariableSpec 
`рвљЉх>
Њ "
 
#__inference__update_step_xla_242721f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`РГљЉх>
Њ "
 Ѕ
#__inference__update_step_xla_242726~xЂu
nЂk
!
gradient@@
<9	%Ђ"
њ@@

p
` VariableSpec 
` љЉх>
Њ "
 
#__inference__update_step_xla_242731f`Ђ]
VЂS

gradient@
0-	Ђ
њ@

p
` VariableSpec 
`рљЉх>
Њ "
 
#__inference__update_step_xla_242736nhЂe
^Ђ[

gradient@0
41	Ђ
њ@0

p
` VariableSpec 
`рЃјЉх>
Њ "
 
#__inference__update_step_xla_242741f`Ђ]
VЂS

gradient0
0-	Ђ
њ0

p
` VariableSpec 
`јЉх>
Њ "
 
#__inference__update_step_xla_242746nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
` ЃѓЋх>
Њ "
 
#__inference__update_step_xla_242751f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`РќЁх>
Њ "
 Э
!__inference__wrapped_model_243461Ї,-./01*+hЂe
^Ђ[
YV
)&
input_7џџџџџџџџџ
)&
input_8џџџџџџџџџ
Њ "1Њ.
,
dense_5!
dense_5џџџџџџџџџЛ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_243958s,-7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
)__inference_conv2d_4_layer_call_fn_243947h,-7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ ")&
unknownџџџџџџџџџ@Л
D__inference_conv2d_5_layer_call_and_return_conditional_losses_244015s./7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
)__inference_conv2d_5_layer_call_fn_244004h./7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ ")&
unknownџџџџџџџџџ@Њ
C__inference_dense_4_layer_call_and_return_conditional_losses_244082c01/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ0
 
(__inference_dense_4_layer_call_fn_244072X01/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ0Њ
C__inference_dense_5_layer_call_and_return_conditional_losses_243938c*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_5_layer_call_fn_243927X*+/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџМ
E__inference_dropout_4_layer_call_and_return_conditional_losses_243990s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 М
E__inference_dropout_4_layer_call_and_return_conditional_losses_243995s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
*__inference_dropout_4_layer_call_fn_243973h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ ")&
unknownџџџџџџџџџ@
*__inference_dropout_4_layer_call_fn_243978h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ ")&
unknownџџџџџџџџџ@М
E__inference_dropout_5_layer_call_and_return_conditional_losses_244047s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 М
E__inference_dropout_5_layer_call_and_return_conditional_losses_244052s;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ "4Ђ1
*'
tensor_0џџџџџџџџџ@
 
*__inference_dropout_5_layer_call_fn_244030h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p
Њ ")&
unknownџџџџџџџџџ@
*__inference_dropout_5_layer_call_fn_244035h;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ@
p 
Њ ")&
unknownџџџџџџџџџ@ц
V__inference_global_average_pooling2d_2_layer_call_and_return_conditional_losses_244063RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "5Ђ2
+(
tensor_0џџџџџџџџџџџџџџџџџџ
 Р
;__inference_global_average_pooling2d_2_layer_call_fn_244057RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*'
unknownџџџџџџџџџџџџџџџџџџл
D__inference_lambda_2_layer_call_and_return_conditional_losses_243904bЂ_
XЂU
KH
"
inputs_0џџџџџџџџџ0
"
inputs_1џџџџџџџџџ0

 
p
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 л
D__inference_lambda_2_layer_call_and_return_conditional_losses_243918bЂ_
XЂU
KH
"
inputs_0џџџџџџџџџ0
"
inputs_1џџџџџџџџџ0

 
p 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Е
)__inference_lambda_2_layer_call_fn_243884bЂ_
XЂU
KH
"
inputs_0џџџџџџџџџ0
"
inputs_1џџџџџџџџџ0

 
p
Њ "!
unknownџџџџџџџџџЕ
)__inference_lambda_2_layer_call_fn_243890bЂ_
XЂU
KH
"
inputs_0џџџџџџџџџ0
"
inputs_1џџџџџџџџџ0

 
p 
Њ "!
unknownџџџџџџџџџѕ
K__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_243968ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
0__inference_max_pooling2d_4_layer_call_fn_243963RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџѕ
K__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_244025ЅRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "OЂL
EB
tensor_04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Я
0__inference_max_pooling2d_5_layer_call_fn_244020RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "DA
unknown4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџП
C__inference_model_4_layer_call_and_return_conditional_losses_243572x,-./01@Ђ=
6Ђ3
)&
input_9џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ0
 П
C__inference_model_4_layer_call_and_return_conditional_losses_243606x,-./01@Ђ=
6Ђ3
)&
input_9џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ0
 
(__inference_model_4_layer_call_fn_243623m,-./01@Ђ=
6Ђ3
)&
input_9џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџ0
(__inference_model_4_layer_call_fn_243640m,-./01@Ђ=
6Ђ3
)&
input_9џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ0ђ
C__inference_model_5_layer_call_and_return_conditional_losses_243743Њ,-./01*+pЂm
fЂc
YV
)&
input_7џџџџџџџџџ
)&
input_8џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ђ
C__inference_model_5_layer_call_and_return_conditional_losses_243788Њ,-./01*+pЂm
fЂc
YV
)&
input_7џџџџџџџџџ
)&
input_8џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Ь
(__inference_model_5_layer_call_fn_243810,-./01*+pЂm
fЂc
YV
)&
input_7џџџџџџџџџ
)&
input_8џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџЬ
(__inference_model_5_layer_call_fn_243832,-./01*+pЂm
fЂc
YV
)&
input_7џџџџџџџџџ
)&
input_8џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџс
$__inference_signature_wrapper_243878И,-./01*+yЂv
Ђ 
oЊl
4
input_7)&
input_7џџџџџџџџџ
4
input_8)&
input_8џџџџџџџџџ"1Њ.
,
dense_5!
dense_5џџџџџџџџџ