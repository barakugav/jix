use crate::ops::common::define_op2;

define_op2!(equal, Equal);
define_op2!(not_equal, NotEqual);
define_op2!(greater, Greater);
define_op2!(greater_equal, GreaterEqual);
define_op2!(less, Less);
define_op2!(less_equal, LessEqual);

define_op2!(maximum, Maximum);
define_op2!(minimum, Minimum);
