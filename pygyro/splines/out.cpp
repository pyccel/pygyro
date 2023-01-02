#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/float.hpp>
#include <pythonic/include/types/numpy_texpr.hpp>
#include <pythonic/include/types/int.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/float64.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/numpy_texpr.hpp>
#include <pythonic/types/float.hpp>
#include <pythonic/types/float64.hpp>
#include <pythonic/types/int.hpp>
#include <pythonic/include/builtins/None.hpp>
#include <pythonic/include/builtins/enumerate.hpp>
#include <pythonic/include/builtins/int_.hpp>
#include <pythonic/include/builtins/len.hpp>
#include <pythonic/include/builtins/pythran/and_.hpp>
#include <pythonic/include/builtins/range.hpp>
#include <pythonic/include/builtins/tuple.hpp>
#include <pythonic/include/numpy/array.hpp>
#include <pythonic/include/numpy/float64.hpp>
#include <pythonic/include/numpy/square.hpp>
#include <pythonic/include/operator_/add.hpp>
#include <pythonic/include/operator_/div.hpp>
#include <pythonic/include/operator_/eq.hpp>
#include <pythonic/include/operator_/iadd.hpp>
#include <pythonic/include/operator_/mul.hpp>
#include <pythonic/include/operator_/sub.hpp>
#include <pythonic/include/types/slice.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/builtins/None.hpp>
#include <pythonic/builtins/enumerate.hpp>
#include <pythonic/builtins/int_.hpp>
#include <pythonic/builtins/len.hpp>
#include <pythonic/builtins/pythran/and_.hpp>
#include <pythonic/builtins/range.hpp>
#include <pythonic/builtins/tuple.hpp>
#include <pythonic/numpy/array.hpp>
#include <pythonic/numpy/float64.hpp>
#include <pythonic/numpy/square.hpp>
#include <pythonic/operator_/add.hpp>
#include <pythonic/operator_/div.hpp>
#include <pythonic/operator_/eq.hpp>
#include <pythonic/operator_/iadd.hpp>
#include <pythonic/operator_/mul.hpp>
#include <pythonic/operator_/sub.hpp>
#include <pythonic/types/slice.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran_out
{
  struct cu_basis_funs_1st_der
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef __type0 __type1;
      typedef __type1 __type2;
      typedef long __type3;
      typedef decltype(pythonic::operator_::sub(std::declval<__type3>(), std::declval<__type1>())) __type5;
      typedef typename pythonic::assignable<__type5>::type __type6;
      typedef __type6 __type7;
      typedef decltype(pythonic::operator_::mul(std::declval<__type3>(), std::declval<__type7>())) __type8;
      typedef decltype(pythonic::operator_::mul(std::declval<__type8>(), std::declval<__type7>())) __type10;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type11;
      typedef __type11 __type12;
      typedef decltype(pythonic::operator_::div(std::declval<__type3>(), std::declval<__type12>())) __type13;
      typedef typename pythonic::assignable<__type13>::type __type14;
      typedef __type14 __type15;
      typedef decltype(pythonic::operator_::mul(std::declval<__type10>(), std::declval<__type15>())) __type16;
      typedef __type16 __type17;
      typedef pythonic::types::none_type __type19;
      typedef typename pythonic::returnable<__type19>::type __type20;
      typedef __type2 __ptype0;
      typedef __type17 __ptype1;
      typedef __type17 __ptype2;
      typedef __type20 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type operator()(argument_type0&& span, argument_type1&& offset, argument_type2&& dx, argument_type3&& ders) const
    ;
  }  ;
  struct cu_basis_funs
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef __type0 __type1;
      typedef __type1 __type2;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::square{})>::type>::type __type3;
      typedef long __type4;
      typedef decltype(pythonic::operator_::sub(std::declval<__type4>(), std::declval<__type1>())) __type6;
      typedef typename pythonic::assignable<__type6>::type __type7;
      typedef __type7 __type8;
      typedef decltype(std::declval<__type3>()(std::declval<__type8>())) __type9;
      typedef decltype(pythonic::operator_::mul(std::declval<__type9>(), std::declval<__type8>())) __type11;
      typedef __type11 __type12;
      typedef pythonic::types::none_type __type14;
      typedef typename pythonic::returnable<__type14>::type __type15;
      typedef __type2 __ptype21;
      typedef __type12 __ptype22;
      typedef __type12 __ptype23;
      typedef __type15 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    inline
    typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0&& span, argument_type1&& offset, argument_type2&& values) const
    ;
  }  ;
  struct cu_find_span
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::int_{})>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type1;
      typedef __type1 __type2;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type3;
      typedef __type3 __type4;
      typedef decltype(pythonic::operator_::sub(std::declval<__type2>(), std::declval<__type4>())) __type5;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type6;
      typedef __type6 __type7;
      typedef decltype(pythonic::operator_::div(std::declval<__type5>(), std::declval<__type7>())) __type8;
      typedef typename pythonic::assignable<__type8>::type __type9;
      typedef __type9 __type10;
      typedef decltype(std::declval<__type0>()(std::declval<__type10>())) __type11;
      typedef typename pythonic::assignable<__type11>::type __type12;
      typedef __type12 __type13;
      typedef decltype(pythonic::operator_::sub(std::declval<__type10>(), std::declval<__type13>())) __type16;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type13>(), std::declval<__type16>())) __type17;
      typedef typename pythonic::returnable<__type17>::type __type18;
      typedef __type18 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
    inline
    typename type<argument_type0, argument_type1, argument_type2>::result_type operator()(argument_type0&& xmin, argument_type1&& dx, argument_type2&& x) const
    ;
  }  ;
  struct cu_eval_spline_2d_vector_11
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef pythonic::types::none_type __type22;
      typedef typename pythonic::returnable<__type22>::type __type23;
      typedef __type3 __ptype42;
      typedef __type9 __ptype43;
      typedef __type13 __ptype44;
      typedef __type18 __ptype45;
      typedef __type20 __ptype46;
      typedef __type20 __ptype47;
      typedef __type23 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_vector_00
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef pythonic::types::none_type __type22;
      typedef typename pythonic::returnable<__type22>::type __type23;
      typedef __type3 __ptype63;
      typedef __type9 __ptype64;
      typedef __type13 __ptype65;
      typedef __type18 __ptype66;
      typedef __type20 __ptype67;
      typedef __type20 __ptype68;
      typedef __type23 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_cross_11
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type21;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
      typedef __type22 __type23;
      typedef decltype(std::declval<__type21>()(std::declval<__type23>())) __type24;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type24>::type::iterator>::value_type>::type __type25;
      typedef __type25 __type26;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type26>::type>::type __type27;
      typedef typename pythonic::lazy<__type27>::type __type28;
      typedef __type28 __type29;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type30;
      typedef __type30 __type31;
      typedef decltype(std::declval<__type21>()(std::declval<__type31>())) __type32;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
      typedef __type33 __type34;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
      typedef typename pythonic::lazy<__type35>::type __type36;
      typedef __type36 __type37;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type29>(), std::declval<__type37>())) __type38;
      typedef __type38 __type39;
      typedef pythonic::types::none_type __type40;
      typedef typename pythonic::returnable<__type40>::type __type41;
      typedef __type3 __ptype84;
      typedef __type9 __ptype85;
      typedef __type13 __ptype86;
      typedef __type18 __ptype87;
      typedef __type20 __ptype88;
      typedef __type39 __ptype89;
      typedef __type41 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
    ;
  }  ;
  struct cu_eval_spline_2d_cross_00
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type21;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
      typedef __type22 __type23;
      typedef decltype(std::declval<__type21>()(std::declval<__type23>())) __type24;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type24>::type::iterator>::value_type>::type __type25;
      typedef __type25 __type26;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type26>::type>::type __type27;
      typedef typename pythonic::lazy<__type27>::type __type28;
      typedef __type28 __type29;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type30;
      typedef __type30 __type31;
      typedef decltype(std::declval<__type21>()(std::declval<__type31>())) __type32;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
      typedef __type33 __type34;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
      typedef typename pythonic::lazy<__type35>::type __type36;
      typedef __type36 __type37;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type29>(), std::declval<__type37>())) __type38;
      typedef __type38 __type39;
      typedef pythonic::types::none_type __type40;
      typedef typename pythonic::returnable<__type40>::type __type41;
      typedef __type3 __ptype92;
      typedef __type9 __ptype93;
      typedef __type13 __ptype94;
      typedef __type18 __ptype95;
      typedef __type20 __ptype96;
      typedef __type39 __ptype97;
      typedef __type41 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
    ;
  }  ;
  struct cu_eval_spline_2d_vector_10
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef pythonic::types::none_type __type22;
      typedef typename pythonic::returnable<__type22>::type __type23;
      typedef __type3 __ptype112;
      typedef __type9 __ptype113;
      typedef __type13 __ptype114;
      typedef __type18 __ptype115;
      typedef __type20 __ptype116;
      typedef __type20 __ptype117;
      typedef __type23 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_vector_01
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef pythonic::types::none_type __type22;
      typedef typename pythonic::returnable<__type22>::type __type23;
      typedef __type3 __ptype133;
      typedef __type9 __ptype134;
      typedef __type13 __ptype135;
      typedef __type18 __ptype136;
      typedef __type20 __ptype137;
      typedef __type20 __ptype138;
      typedef __type23 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_cross_10
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type21;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
      typedef __type22 __type23;
      typedef decltype(std::declval<__type21>()(std::declval<__type23>())) __type24;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type24>::type::iterator>::value_type>::type __type25;
      typedef __type25 __type26;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type26>::type>::type __type27;
      typedef typename pythonic::lazy<__type27>::type __type28;
      typedef __type28 __type29;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type30;
      typedef __type30 __type31;
      typedef decltype(std::declval<__type21>()(std::declval<__type31>())) __type32;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
      typedef __type33 __type34;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
      typedef typename pythonic::lazy<__type35>::type __type36;
      typedef __type36 __type37;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type29>(), std::declval<__type37>())) __type38;
      typedef __type38 __type39;
      typedef pythonic::types::none_type __type40;
      typedef typename pythonic::returnable<__type40>::type __type41;
      typedef __type3 __ptype154;
      typedef __type9 __ptype155;
      typedef __type13 __ptype156;
      typedef __type18 __ptype157;
      typedef __type20 __ptype158;
      typedef __type39 __ptype159;
      typedef __type41 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
    ;
  }  ;
  struct cu_eval_spline_2d_cross_01
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type10;
      typedef __type10 __type11;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type11>::type>::type __type12;
      typedef __type12 __type13;
      typedef indexable_container<__type4, typename std::remove_reference<__type12>::type> __type14;
      typedef typename __combined<__type10,__type14>::type __type15;
      typedef __type15 __type16;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
      typedef __type17 __type18;
      typedef double __type19;
      typedef __type19 __type20;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type21;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
      typedef __type22 __type23;
      typedef decltype(std::declval<__type21>()(std::declval<__type23>())) __type24;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type24>::type::iterator>::value_type>::type __type25;
      typedef __type25 __type26;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type26>::type>::type __type27;
      typedef typename pythonic::lazy<__type27>::type __type28;
      typedef __type28 __type29;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type30;
      typedef __type30 __type31;
      typedef decltype(std::declval<__type21>()(std::declval<__type31>())) __type32;
      typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type32>::type::iterator>::value_type>::type __type33;
      typedef __type33 __type34;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
      typedef typename pythonic::lazy<__type35>::type __type36;
      typedef __type36 __type37;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type29>(), std::declval<__type37>())) __type38;
      typedef __type38 __type39;
      typedef pythonic::types::none_type __type40;
      typedef typename pythonic::returnable<__type40>::type __type41;
      typedef __type3 __ptype174;
      typedef __type9 __ptype175;
      typedef __type13 __ptype176;
      typedef __type18 __ptype177;
      typedef __type20 __ptype178;
      typedef __type39 __ptype179;
      typedef __type41 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
    ;
  }  ;
  struct cu_eval_spline_2d_scalar
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 = long, typename argument_type8 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef double __type4;
      typedef typename pythonic::assignable<__type4>::type __type5;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type6;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type4>(), std::declval<__type4>(), std::declval<__type4>(), std::declval<__type4>())) __type7;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type7>(), std::declval<__type7>(), std::declval<__type7>(), std::declval<__type7>())) __type8;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type9;
      typedef decltype(std::declval<__type6>()(std::declval<__type8>(), std::declval<__type9>())) __type10;
      typedef typename pythonic::assignable<__type10>::type __type11;
      typedef long __type12;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type12>(), std::declval<__type12>())) __type13;
      typedef indexable<__type13> __type14;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type15;
      typedef __type15 __type16;
      typedef pythonic::types::contiguous_slice __type17;
      typedef decltype(std::declval<__type16>()(std::declval<__type17>(), std::declval<__type17>())) __type18;
      typedef container<typename std::remove_reference<__type18>::type> __type19;
      typedef typename __combined<__type11,__type19>::type __type20;
      typedef __type20 __type21;
      typedef decltype(std::declval<__type21>()[std::declval<__type13>()]) __type22;
      typedef decltype(std::declval<__type6>()(std::declval<__type7>(), std::declval<__type9>())) __type23;
      typedef typename pythonic::assignable<__type23>::type __type24;
      typedef std::integral_constant<long,0> __type25;
      typedef cu_find_span __type26;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type27;
      typedef __type27 __type28;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type28>::type>::type __type29;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type28>::type>::type __type30;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type31;
      typedef __type31 __type32;
      typedef decltype(std::declval<__type26>()(std::declval<__type29>(), std::declval<__type30>(), std::declval<__type32>())) __type33;
      typedef typename pythonic::assignable<__type33>::type __type34;
      typedef __type34 __type35;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type35>::type>::type __type36;
      typedef typename pythonic::assignable<__type36>::type __type37;
      typedef __type37 __type38;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type35>::type>::type __type39;
      typedef __type24 __type40;
      typedef typename cu_basis_funs::type<__type38, __type39, __type40>::__ptype22 __type41;
      typedef indexable_container<__type25, typename std::remove_reference<__type41>::type> __type42;
      typedef typename __combined<__type40,__type42>::type __type43;
      typedef typename cu_basis_funs::type<__type38, __type39, __type43>::__ptype23 __type44;
      typedef indexable_container<__type25, typename std::remove_reference<__type44>::type> __type45;
      typedef std::integral_constant<long,1> __type47;
      typedef typename cu_basis_funs::type<__type38, __type39, __type40>::__ptype21 __type48;
      typedef indexable_container<__type47, typename std::remove_reference<__type48>::type> __type49;
      typedef typename __combined<__type34,__type49>::type __type50;
      typedef __type50 __type51;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type51>::type>::type __type52;
      typedef typename __combined<__type39,__type52>::type __type53;
      typedef typename pythonic::assignable<__type2>::type __type54;
      typedef __type54 __type55;
      typedef typename __combined<__type24,__type42,__type45,__type40>::type __type56;
      typedef __type56 __type57;
      typedef typename cu_basis_funs_1st_der::type<__type38, __type53, __type55, __type57>::__ptype1 __type58;
      typedef indexable_container<__type25, typename std::remove_reference<__type58>::type> __type59;
      typedef typename __combined<__type57,__type59>::type __type60;
      typedef typename cu_basis_funs_1st_der::type<__type38, __type53, __type55, __type60>::__ptype2 __type61;
      typedef indexable_container<__type25, typename std::remove_reference<__type61>::type> __type62;
      typedef typename __combined<__type24,__type42,__type45,__type59,__type62,__type40,__type57>::type __type63;
      typedef __type63 __type64;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type64>::type>::type __type65;
      typedef decltype(pythonic::operator_::mul(std::declval<__type22>(), std::declval<__type65>())) __type66;
      typedef container<typename std::remove_reference<__type66>::type> __type67;
      typedef typename __combined<__type11,__type14,__type19,__type67>::type __type68;
      typedef __type68 __type69;
      typedef decltype(std::declval<__type69>()[std::declval<__type13>()]) __type70;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type64>::type>::type __type72;
      typedef decltype(pythonic::operator_::mul(std::declval<__type70>(), std::declval<__type72>())) __type73;
      typedef container<typename std::remove_reference<__type73>::type> __type74;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74>::type __type75;
      typedef __type75 __type76;
      typedef decltype(std::declval<__type76>()[std::declval<__type13>()]) __type77;
      typedef typename std::tuple_element<2,typename std::remove_reference<__type64>::type>::type __type79;
      typedef decltype(pythonic::operator_::mul(std::declval<__type77>(), std::declval<__type79>())) __type80;
      typedef container<typename std::remove_reference<__type80>::type> __type81;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81>::type __type82;
      typedef __type82 __type83;
      typedef decltype(std::declval<__type83>()[std::declval<__type13>()]) __type84;
      typedef typename std::tuple_element<3,typename std::remove_reference<__type64>::type>::type __type86;
      typedef decltype(pythonic::operator_::mul(std::declval<__type84>(), std::declval<__type86>())) __type87;
      typedef container<typename std::remove_reference<__type87>::type> __type88;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88>::type __type89;
      typedef __type89 __type90;
      typedef decltype(std::declval<__type90>()[std::declval<__type13>()]) __type91;
      typedef indexable_container<__type47, typename std::remove_reference<__type2>::type> __type92;
      typedef typename __combined<__type0,__type92>::type __type93;
      typedef __type93 __type94;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type94>::type>::type __type95;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type97;
      typedef __type97 __type98;
      typedef decltype(std::declval<__type26>()(std::declval<__type95>(), std::declval<__type55>(), std::declval<__type98>())) __type99;
      typedef typename pythonic::assignable<__type99>::type __type100;
      typedef __type100 __type101;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type101>::type>::type __type102;
      typedef typename pythonic::assignable<__type102>::type __type103;
      typedef __type103 __type104;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type101>::type>::type __type105;
      typedef typename cu_basis_funs::type<__type104, __type105, __type40>::__ptype22 __type107;
      typedef indexable_container<__type25, typename std::remove_reference<__type107>::type> __type108;
      typedef typename __combined<__type40,__type108>::type __type109;
      typedef typename cu_basis_funs::type<__type104, __type105, __type109>::__ptype23 __type110;
      typedef indexable_container<__type25, typename std::remove_reference<__type110>::type> __type111;
      typedef typename cu_basis_funs::type<__type104, __type105, __type40>::__ptype21 __type113;
      typedef indexable_container<__type47, typename std::remove_reference<__type113>::type> __type114;
      typedef typename __combined<__type100,__type114>::type __type115;
      typedef __type115 __type116;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type116>::type>::type __type117;
      typedef typename __combined<__type105,__type117>::type __type118;
      typedef typename __combined<__type24,__type108,__type111,__type40>::type __type120;
      typedef __type120 __type121;
      typedef typename cu_basis_funs_1st_der::type<__type104, __type118, __type55, __type121>::__ptype1 __type122;
      typedef indexable_container<__type25, typename std::remove_reference<__type122>::type> __type123;
      typedef typename __combined<__type121,__type123>::type __type124;
      typedef typename cu_basis_funs_1st_der::type<__type104, __type118, __type55, __type124>::__ptype2 __type125;
      typedef indexable_container<__type25, typename std::remove_reference<__type125>::type> __type126;
      typedef typename __combined<__type24,__type108,__type111,__type123,__type126,__type40,__type121>::type __type127;
      typedef __type127 __type128;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type128>::type>::type __type129;
      typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type129>())) __type130;
      typedef decltype(pythonic::operator_::mul(std::declval<__type91>(), std::declval<__type65>())) __type135;
      typedef container<typename std::remove_reference<__type135>::type> __type136;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136>::type __type137;
      typedef __type137 __type138;
      typedef decltype(std::declval<__type138>()[std::declval<__type13>()]) __type139;
      typedef decltype(pythonic::operator_::mul(std::declval<__type139>(), std::declval<__type72>())) __type142;
      typedef container<typename std::remove_reference<__type142>::type> __type143;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143>::type __type144;
      typedef __type144 __type145;
      typedef decltype(std::declval<__type145>()[std::declval<__type13>()]) __type146;
      typedef decltype(pythonic::operator_::mul(std::declval<__type146>(), std::declval<__type79>())) __type149;
      typedef container<typename std::remove_reference<__type149>::type> __type150;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150>::type __type151;
      typedef __type151 __type152;
      typedef decltype(std::declval<__type152>()[std::declval<__type13>()]) __type153;
      typedef decltype(pythonic::operator_::mul(std::declval<__type153>(), std::declval<__type86>())) __type156;
      typedef container<typename std::remove_reference<__type156>::type> __type157;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157>::type __type158;
      typedef __type158 __type159;
      typedef decltype(std::declval<__type159>()[std::declval<__type13>()]) __type160;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type128>::type>::type __type162;
      typedef decltype(pythonic::operator_::mul(std::declval<__type160>(), std::declval<__type162>())) __type163;
      typedef decltype(pythonic::operator_::mul(std::declval<__type160>(), std::declval<__type65>())) __type168;
      typedef container<typename std::remove_reference<__type168>::type> __type169;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169>::type __type170;
      typedef __type170 __type171;
      typedef decltype(std::declval<__type171>()[std::declval<__type13>()]) __type172;
      typedef decltype(pythonic::operator_::mul(std::declval<__type172>(), std::declval<__type72>())) __type175;
      typedef container<typename std::remove_reference<__type175>::type> __type176;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176>::type __type177;
      typedef __type177 __type178;
      typedef decltype(std::declval<__type178>()[std::declval<__type13>()]) __type179;
      typedef decltype(pythonic::operator_::mul(std::declval<__type179>(), std::declval<__type79>())) __type182;
      typedef container<typename std::remove_reference<__type182>::type> __type183;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183>::type __type184;
      typedef __type184 __type185;
      typedef decltype(std::declval<__type185>()[std::declval<__type13>()]) __type186;
      typedef decltype(pythonic::operator_::mul(std::declval<__type186>(), std::declval<__type86>())) __type189;
      typedef container<typename std::remove_reference<__type189>::type> __type190;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183,__type190>::type __type191;
      typedef __type191 __type192;
      typedef decltype(std::declval<__type192>()[std::declval<__type13>()]) __type193;
      typedef typename std::tuple_element<2,typename std::remove_reference<__type128>::type>::type __type195;
      typedef decltype(pythonic::operator_::mul(std::declval<__type193>(), std::declval<__type195>())) __type196;
      typedef decltype(pythonic::operator_::mul(std::declval<__type193>(), std::declval<__type65>())) __type201;
      typedef container<typename std::remove_reference<__type201>::type> __type202;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183,__type190,__type202>::type __type203;
      typedef __type203 __type204;
      typedef decltype(std::declval<__type204>()[std::declval<__type13>()]) __type205;
      typedef decltype(pythonic::operator_::mul(std::declval<__type205>(), std::declval<__type72>())) __type208;
      typedef container<typename std::remove_reference<__type208>::type> __type209;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183,__type190,__type202,__type209>::type __type210;
      typedef __type210 __type211;
      typedef decltype(std::declval<__type211>()[std::declval<__type13>()]) __type212;
      typedef decltype(pythonic::operator_::mul(std::declval<__type212>(), std::declval<__type79>())) __type215;
      typedef container<typename std::remove_reference<__type215>::type> __type216;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183,__type190,__type202,__type209,__type216>::type __type217;
      typedef __type217 __type218;
      typedef decltype(std::declval<__type218>()[std::declval<__type13>()]) __type219;
      typedef decltype(pythonic::operator_::mul(std::declval<__type219>(), std::declval<__type86>())) __type222;
      typedef container<typename std::remove_reference<__type222>::type> __type223;
      typedef typename __combined<__type11,__type14,__type19,__type67,__type74,__type81,__type88,__type136,__type143,__type150,__type157,__type169,__type176,__type183,__type190,__type202,__type209,__type216,__type223>::type __type224;
      typedef __type224 __type225;
      typedef decltype(std::declval<__type225>()[std::declval<__type13>()]) __type226;
      typedef typename std::tuple_element<3,typename std::remove_reference<__type128>::type>::type __type228;
      typedef decltype(pythonic::operator_::mul(std::declval<__type226>(), std::declval<__type228>())) __type229;
      typedef typename __combined<__type5,__type130,__type163,__type196,__type229>::type __type230;
      typedef __type230 __type231;
      typedef typename pythonic::returnable<__type231>::type __type232;
      typedef __type3 __ptype194;
      typedef __type232 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 = long, typename argument_type8 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7 der1= 0L, argument_type8 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_1d_vector
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef std::integral_constant<long,0> __type4;
      typedef indexable_container<__type4, typename std::remove_reference<__type2>::type> __type5;
      typedef typename __combined<__type0,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type7>::type>::type __type8;
      typedef __type8 __type9;
      typedef double __type10;
      typedef __type10 __type11;
      typedef pythonic::types::none_type __type13;
      typedef typename pythonic::returnable<__type13>::type __type14;
      typedef __type3 __ptype195;
      typedef __type9 __ptype196;
      typedef __type11 __ptype197;
      typedef __type11 __ptype198;
      typedef __type14 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5>::result_type operator()(argument_type0&& x, argument_type1&& knots, argument_type2&& degree, argument_type3&& coeffs, argument_type4&& y, argument_type5 der= 0L) const
    ;
  }  ;
  struct cu_eval_spline_1d_scalar
  {
    typedef void callable;
    typedef void pure;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef double __type4;
      typedef typename pythonic::assignable<__type4>::type __type5;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type6;
      typedef __type6 __type7;
      typedef cu_find_span __type8;
      typedef std::integral_constant<long,1> __type9;
      typedef indexable_container<__type9, typename std::remove_reference<__type2>::type> __type10;
      typedef typename __combined<__type0,__type10>::type __type11;
      typedef __type11 __type12;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type12>::type>::type __type13;
      typedef typename pythonic::assignable<__type2>::type __type14;
      typedef __type14 __type15;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type16;
      typedef __type16 __type17;
      typedef decltype(std::declval<__type8>()(std::declval<__type13>(), std::declval<__type15>(), std::declval<__type17>())) __type18;
      typedef typename pythonic::assignable<__type18>::type __type19;
      typedef __type19 __type20;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type20>::type>::type __type21;
      typedef typename pythonic::assignable<__type21>::type __type22;
      typedef __type22 __type23;
      typedef long __type24;
      typedef decltype(pythonic::operator_::sub(std::declval<__type23>(), std::declval<__type24>())) __type25;
      typedef decltype(pythonic::operator_::add(std::declval<__type25>(), std::declval<__type24>())) __type26;
      typedef decltype(std::declval<__type7>()[std::declval<__type26>()]) __type27;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type28;
      typedef decltype(pythonic::types::make_tuple(std::declval<__type4>(), std::declval<__type4>(), std::declval<__type4>(), std::declval<__type4>())) __type29;
      typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type30;
      typedef decltype(std::declval<__type28>()(std::declval<__type29>(), std::declval<__type30>())) __type31;
      typedef typename pythonic::assignable<__type31>::type __type32;
      typedef std::integral_constant<long,0> __type33;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type20>::type>::type __type35;
      typedef __type32 __type36;
      typedef typename cu_basis_funs::type<__type23, __type35, __type36>::__ptype22 __type37;
      typedef indexable_container<__type33, typename std::remove_reference<__type37>::type> __type38;
      typedef typename __combined<__type36,__type38>::type __type39;
      typedef typename cu_basis_funs::type<__type23, __type35, __type39>::__ptype23 __type40;
      typedef indexable_container<__type33, typename std::remove_reference<__type40>::type> __type41;
      typedef typename cu_basis_funs::type<__type23, __type35, __type36>::__ptype21 __type43;
      typedef indexable_container<__type9, typename std::remove_reference<__type43>::type> __type44;
      typedef typename __combined<__type19,__type44>::type __type45;
      typedef __type45 __type46;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type46>::type>::type __type47;
      typedef typename __combined<__type35,__type47>::type __type48;
      typedef typename __combined<__type32,__type38,__type41,__type36>::type __type50;
      typedef __type50 __type51;
      typedef typename cu_basis_funs_1st_der::type<__type23, __type48, __type15, __type51>::__ptype1 __type52;
      typedef indexable_container<__type33, typename std::remove_reference<__type52>::type> __type53;
      typedef typename __combined<__type51,__type53>::type __type54;
      typedef typename cu_basis_funs_1st_der::type<__type23, __type48, __type15, __type54>::__ptype2 __type55;
      typedef indexable_container<__type33, typename std::remove_reference<__type55>::type> __type56;
      typedef typename __combined<__type32,__type38,__type41,__type53,__type56,__type36,__type51>::type __type57;
      typedef __type57 __type58;
      typedef typename std::tuple_element<0,typename std::remove_reference<__type58>::type>::type __type59;
      typedef decltype(pythonic::operator_::mul(std::declval<__type27>(), std::declval<__type59>())) __type60;
      typedef typename std::tuple_element<1,typename std::remove_reference<__type58>::type>::type __type67;
      typedef decltype(pythonic::operator_::mul(std::declval<__type27>(), std::declval<__type67>())) __type68;
      typedef typename std::tuple_element<2,typename std::remove_reference<__type58>::type>::type __type75;
      typedef decltype(pythonic::operator_::mul(std::declval<__type27>(), std::declval<__type75>())) __type76;
      typedef typename std::tuple_element<3,typename std::remove_reference<__type58>::type>::type __type83;
      typedef decltype(pythonic::operator_::mul(std::declval<__type27>(), std::declval<__type83>())) __type84;
      typedef typename __combined<__type5,__type60,__type68,__type76,__type84>::type __type85;
      typedef __type85 __type86;
      typedef typename pythonic::returnable<__type86>::type __type87;
      typedef __type3 __ptype231;
      typedef __type87 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type operator()(argument_type0&& x, argument_type1&& knots, argument_type2&& degree, argument_type3&& coeffs, argument_type4 der= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_vector
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
      typedef __type4 __type5;
      typedef typename __combined<__type4,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename __combined<__type4,__type5,__type7>::type __type8;
      typedef __type8 __type9;
      typedef typename __combined<__type4,__type5,__type7,__type9>::type __type10;
      typedef __type10 __type11;
      typedef typename __combined<__type4,__type5,__type7,__type9,__type11>::type __type12;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type13;
      typedef __type13 __type14;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type15;
      typedef __type15 __type16;
      typedef typename __combined<__type15,__type16>::type __type17;
      typedef __type17 __type18;
      typedef typename __combined<__type15,__type16,__type18>::type __type19;
      typedef __type19 __type20;
      typedef typename __combined<__type15,__type16,__type18,__type20>::type __type21;
      typedef __type21 __type22;
      typedef typename __combined<__type15,__type16,__type18,__type20,__type22>::type __type23;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type5>::type>::type __type24;
      typedef __type24 __type25;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type26;
      typedef __type26 __type27;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type7>::type>::type __type28;
      typedef __type28 __type29;
      typedef typename __combined<__type28,__type29>::type __type30;
      typedef __type30 __type31;
      typedef typename __combined<__type28,__type29,__type31>::type __type32;
      typedef __type32 __type33;
      typedef typename __combined<__type28,__type29,__type31,__type33>::type __type34;
      typedef __type34 __type35;
      typedef typename __combined<__type28,__type29,__type31,__type33,__type35>::type __type36;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type8>::type>::type __type37;
      typedef __type37 __type38;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type9>::type>::type __type39;
      typedef __type39 __type40;
      typedef typename cu_eval_spline_2d_vector_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29, __type38, __type40>::__ptype63 __type41;
      typedef __type41 __type42;
      typedef typename cu_eval_spline_2d_vector_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29, __type38, __type40>::__ptype65 __type44;
      typedef __type44 __type45;
      typedef typename cu_eval_spline_2d_vector_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29, __type38, __type40>::__ptype67 __type47;
      typedef __type47 __type48;
      typedef pythonic::types::none_type __type50;
      typedef typename pythonic::returnable<__type50>::type __type51;
      typedef __type42 __ptype232;
      typedef __type42 __ptype233;
      typedef __type45 __ptype236;
      typedef __type45 __ptype237;
      typedef __type48 __ptype240;
      typedef __type48 __ptype241;
      typedef __type51 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  struct cu_eval_spline_2d_cross
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef __type0 __type1;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type2;
      typedef __type2 __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
      typedef __type4 __type5;
      typedef typename __combined<__type4,__type5>::type __type6;
      typedef __type6 __type7;
      typedef typename __combined<__type4,__type5,__type7>::type __type8;
      typedef __type8 __type9;
      typedef typename __combined<__type4,__type5,__type7,__type9>::type __type10;
      typedef __type10 __type11;
      typedef typename __combined<__type4,__type5,__type7,__type9,__type11>::type __type12;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type13;
      typedef __type13 __type14;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type15;
      typedef __type15 __type16;
      typedef typename __combined<__type15,__type16>::type __type17;
      typedef __type17 __type18;
      typedef typename __combined<__type15,__type16,__type18>::type __type19;
      typedef __type19 __type20;
      typedef typename __combined<__type15,__type16,__type18,__type20>::type __type21;
      typedef __type21 __type22;
      typedef typename __combined<__type15,__type16,__type18,__type20,__type22>::type __type23;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type5>::type>::type __type24;
      typedef __type24 __type25;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type26;
      typedef __type26 __type27;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type7>::type>::type __type28;
      typedef __type28 __type29;
      typedef typename __combined<__type28,__type29>::type __type30;
      typedef __type30 __type31;
      typedef typename __combined<__type28,__type29,__type31>::type __type32;
      typedef __type32 __type33;
      typedef typename __combined<__type28,__type29,__type31,__type33>::type __type34;
      typedef __type34 __type35;
      typedef typename __combined<__type28,__type29,__type31,__type33,__type35>::type __type36;
      typedef typename cu_eval_spline_2d_cross_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29>::__ptype92 __type37;
      typedef __type37 __type38;
      typedef typename cu_eval_spline_2d_cross_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29>::__ptype94 __type40;
      typedef __type40 __type41;
      typedef typename cu_eval_spline_2d_cross_00::type<__type1, __type3, __type5, __type14, __type16, __type25, __type27, __type29>::__ptype96 __type43;
      typedef __type43 __type44;
      typedef pythonic::types::none_type __type46;
      typedef typename pythonic::returnable<__type46>::type __type47;
      typedef __type38 __ptype280;
      typedef __type38 __ptype281;
      typedef __type41 __ptype284;
      typedef __type41 __ptype285;
      typedef __type44 __ptype288;
      typedef __type44 __ptype289;
      typedef __type47 result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 = long, typename argument_type9 = long>
    inline
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1= 0L, argument_type9 der2= 0L) const
    ;
  }  ;
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
  inline
  typename cu_basis_funs_1st_der::type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type cu_basis_funs_1st_der::operator()(argument_type0&& span, argument_type1&& offset, argument_type2&& dx, argument_type3&& ders) const
  {
    typename pythonic::assignable_noescape<decltype(pythonic::operator_::sub(1L, offset))>::type b = pythonic::operator_::sub(1L, offset);
    typename pythonic::assignable_noescape<decltype(offset)>::type o = offset;
    typename pythonic::assignable_noescape<decltype(pythonic::operator_::div(1L, dx))>::type idx = pythonic::operator_::div(1L, dx);
    std::get<0>(ders) = pythonic::operator_::mul(pythonic::operator_::mul(pythonic::operator_::mul(3L, b), b), idx);
    std::get<1>(ders) = pythonic::operator_::mul(pythonic::operator_::mul(pythonic::operator_::mul(3L, b), idx), pythonic::operator_::sub(pythonic::operator_::mul(3L, b), 4L));
    std::get<2>(ders) = pythonic::operator_::mul(pythonic::operator_::mul(pythonic::operator_::mul(3L, o), idx), pythonic::operator_::sub(4L, pythonic::operator_::mul(3L, o)));
    std::get<3>(ders) = pythonic::operator_::mul(pythonic::operator_::mul(pythonic::operator_::mul(3L, o), o), idx);
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
  inline
  typename cu_basis_funs::type<argument_type0, argument_type1, argument_type2>::result_type cu_basis_funs::operator()(argument_type0&& span, argument_type1&& offset, argument_type2&& values) const
  {
    typename pythonic::assignable_noescape<decltype(pythonic::operator_::sub(1L, offset))>::type b = pythonic::operator_::sub(1L, offset);
    typename pythonic::assignable_noescape<decltype(offset)>::type o = offset;
    std::get<0>(values) = pythonic::operator_::mul(pythonic::numpy::functor::square{}(b), b);
    std::get<1>(values) = pythonic::operator_::add(1L, pythonic::operator_::mul(3L, pythonic::operator_::sub(1L, pythonic::operator_::mul(pythonic::numpy::functor::square{}(b), pythonic::operator_::sub(2L, b)))));
    std::get<2>(values) = pythonic::operator_::add(1L, pythonic::operator_::mul(3L, pythonic::operator_::sub(1L, pythonic::operator_::mul(pythonic::numpy::functor::square{}(o), pythonic::operator_::sub(2L, o)))));
    std::get<3>(values) = pythonic::operator_::mul(pythonic::numpy::functor::square{}(o), o);
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 >
  inline
  typename cu_find_span::type<argument_type0, argument_type1, argument_type2>::result_type cu_find_span::operator()(argument_type0&& xmin, argument_type1&& dx, argument_type2&& x) const
  {
    typename pythonic::assignable_noescape<decltype(pythonic::operator_::div(pythonic::operator_::sub(x, xmin), dx))>::type normalised_pos = pythonic::operator_::div(pythonic::operator_::sub(x, xmin), dx);
    typename pythonic::assignable_noescape<decltype(pythonic::builtins::functor::int_{}(normalised_pos))>::type span = pythonic::builtins::functor::int_{}(normalised_pos);
    return pythonic::types::make_tuple(span, pythonic::operator_::sub(normalised_pos, span));
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_vector_11::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_vector_11::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
    typedef __type22 __type23;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type24;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type25;
    typedef decltype(std::declval<__type25>()(std::declval<__type23>())) __type27;
    typedef decltype(std::declval<__type24>()(std::declval<__type27>())) __type28;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type28>::type::iterator>::value_type>::type __type29;
    typedef __type29 __type30;
    typedef decltype(std::declval<__type23>()[std::declval<__type30>()]) __type31;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type31>())) __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
    typedef typename pythonic::assignable<__type35>::type __type36;
    typedef __type36 __type37;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type34>::type>::type __type38;
    typedef __type5 __type40;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type40>::__ptype0 __type41;
    typedef indexable_container<__type14, typename std::remove_reference<__type41>::type> __type42;
    typedef typename __combined<__type33,__type42>::type __type43;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type40>::__ptype1 __type44;
    typedef indexable_container<__type6, typename std::remove_reference<__type44>::type> __type45;
    typedef typename __combined<__type40,__type45>::type __type46;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type46>::__ptype2 __type47;
    typedef indexable_container<__type6, typename std::remove_reference<__type47>::type> __type48;
    typedef typename __combined<__type5,__type45,__type48,__type40>::type __type49;
    typedef typename pythonic::assignable<__type49>::type __type50;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type51;
    typedef __type51 __type52;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type52>::type>::type __type53;
    typedef typename pythonic::assignable<__type53>::type __type54;
    typedef __type54 __type55;
    typedef indexable_container<__type6, typename std::remove_reference<__type53>::type> __type56;
    typedef typename __combined<__type51,__type56>::type __type57;
    typedef __type57 __type58;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type58>::type>::type __type59;
    typedef indexable_container<__type14, typename std::remove_reference<__type59>::type> __type60;
    typedef typename __combined<__type51,__type56,__type60>::type __type61;
    typedef typename pythonic::assignable<__type59>::type __type62;
    typedef __type62 __type63;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type64;
    typedef __type64 __type65;
    typedef decltype(std::declval<__type65>()[std::declval<__type30>()]) __type67;
    typedef decltype(std::declval<__type7>()(std::declval<__type55>(), std::declval<__type63>(), std::declval<__type67>())) __type68;
    typedef typename pythonic::assignable<__type68>::type __type69;
    typedef __type69 __type70;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type70>::type>::type __type71;
    typedef typename pythonic::assignable<__type71>::type __type72;
    typedef __type72 __type73;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type70>::type>::type __type74;
    typedef typename cu_basis_funs_1st_der::type<__type73, __type74, __type63, __type40>::__ptype0 __type77;
    typedef indexable_container<__type14, typename std::remove_reference<__type77>::type> __type78;
    typedef typename __combined<__type69,__type78>::type __type79;
    typedef typename cu_basis_funs_1st_der::type<__type73, __type74, __type63, __type40>::__ptype1 __type80;
    typedef indexable_container<__type6, typename std::remove_reference<__type80>::type> __type81;
    typedef typename __combined<__type40,__type81>::type __type82;
    typedef typename cu_basis_funs_1st_der::type<__type73, __type74, __type63, __type82>::__ptype2 __type83;
    typedef indexable_container<__type6, typename std::remove_reference<__type83>::type> __type84;
    typedef typename __combined<__type5,__type81,__type84,__type40>::type __type85;
    typedef typename pythonic::assignable<__type85>::type __type86;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type87;
    typedef decltype(std::declval<__type0>()(std::declval<__type87>(), std::declval<__type3>())) __type88;
    typedef typename pythonic::assignable<__type88>::type __type89;
    typedef long __type90;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type90>(), std::declval<__type90>())) __type91;
    typedef indexable<__type91> __type92;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type93;
    typedef __type93 __type94;
    typedef pythonic::types::contiguous_slice __type95;
    typedef decltype(std::declval<__type94>()(std::declval<__type95>(), std::declval<__type95>())) __type96;
    typedef container<typename std::remove_reference<__type96>::type> __type97;
    typedef typename __combined<__type89,__type97>::type __type98;
    typedef __type98 __type99;
    typedef decltype(std::declval<__type99>()[std::declval<__type91>()]) __type100;
    typedef __type85 __type101;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type101>::type>::type __type102;
    typedef decltype(pythonic::operator_::mul(std::declval<__type100>(), std::declval<__type102>())) __type103;
    typedef container<typename std::remove_reference<__type103>::type> __type104;
    typedef typename __combined<__type89,__type92,__type97,__type104>::type __type105;
    typedef __type105 __type106;
    typedef decltype(std::declval<__type106>()[std::declval<__type91>()]) __type107;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type101>::type>::type __type109;
    typedef decltype(pythonic::operator_::mul(std::declval<__type107>(), std::declval<__type109>())) __type110;
    typedef container<typename std::remove_reference<__type110>::type> __type111;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111>::type __type112;
    typedef __type112 __type113;
    typedef decltype(std::declval<__type113>()[std::declval<__type91>()]) __type114;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type101>::type>::type __type116;
    typedef decltype(pythonic::operator_::mul(std::declval<__type114>(), std::declval<__type116>())) __type117;
    typedef container<typename std::remove_reference<__type117>::type> __type118;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118>::type __type119;
    typedef __type119 __type120;
    typedef decltype(std::declval<__type120>()[std::declval<__type91>()]) __type121;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type101>::type>::type __type123;
    typedef decltype(pythonic::operator_::mul(std::declval<__type121>(), std::declval<__type123>())) __type124;
    typedef container<typename std::remove_reference<__type124>::type> __type125;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125>::type __type126;
    typedef __type126 __type127;
    typedef decltype(std::declval<__type127>()[std::declval<__type91>()]) __type128;
    typedef decltype(pythonic::operator_::mul(std::declval<__type128>(), std::declval<__type102>())) __type131;
    typedef container<typename std::remove_reference<__type131>::type> __type132;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132>::type __type133;
    typedef __type133 __type134;
    typedef decltype(std::declval<__type134>()[std::declval<__type91>()]) __type135;
    typedef decltype(pythonic::operator_::mul(std::declval<__type135>(), std::declval<__type109>())) __type138;
    typedef container<typename std::remove_reference<__type138>::type> __type139;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139>::type __type140;
    typedef __type140 __type141;
    typedef decltype(std::declval<__type141>()[std::declval<__type91>()]) __type142;
    typedef decltype(pythonic::operator_::mul(std::declval<__type142>(), std::declval<__type116>())) __type145;
    typedef container<typename std::remove_reference<__type145>::type> __type146;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146>::type __type147;
    typedef __type147 __type148;
    typedef decltype(std::declval<__type148>()[std::declval<__type91>()]) __type149;
    typedef decltype(pythonic::operator_::mul(std::declval<__type149>(), std::declval<__type123>())) __type152;
    typedef container<typename std::remove_reference<__type152>::type> __type153;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153>::type __type154;
    typedef __type154 __type155;
    typedef decltype(std::declval<__type155>()[std::declval<__type91>()]) __type156;
    typedef decltype(pythonic::operator_::mul(std::declval<__type156>(), std::declval<__type102>())) __type159;
    typedef container<typename std::remove_reference<__type159>::type> __type160;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160>::type __type161;
    typedef __type161 __type162;
    typedef decltype(std::declval<__type162>()[std::declval<__type91>()]) __type163;
    typedef decltype(pythonic::operator_::mul(std::declval<__type163>(), std::declval<__type109>())) __type166;
    typedef container<typename std::remove_reference<__type166>::type> __type167;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167>::type __type168;
    typedef __type168 __type169;
    typedef decltype(std::declval<__type169>()[std::declval<__type91>()]) __type170;
    typedef decltype(pythonic::operator_::mul(std::declval<__type170>(), std::declval<__type116>())) __type173;
    typedef container<typename std::remove_reference<__type173>::type> __type174;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174>::type __type175;
    typedef __type175 __type176;
    typedef decltype(std::declval<__type176>()[std::declval<__type91>()]) __type177;
    typedef decltype(pythonic::operator_::mul(std::declval<__type177>(), std::declval<__type123>())) __type180;
    typedef container<typename std::remove_reference<__type180>::type> __type181;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181>::type __type182;
    typedef __type182 __type183;
    typedef decltype(std::declval<__type183>()[std::declval<__type91>()]) __type184;
    typedef decltype(pythonic::operator_::mul(std::declval<__type184>(), std::declval<__type102>())) __type187;
    typedef container<typename std::remove_reference<__type187>::type> __type188;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188>::type __type189;
    typedef __type189 __type190;
    typedef decltype(std::declval<__type190>()[std::declval<__type91>()]) __type191;
    typedef decltype(pythonic::operator_::mul(std::declval<__type191>(), std::declval<__type109>())) __type194;
    typedef container<typename std::remove_reference<__type194>::type> __type195;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195>::type __type196;
    typedef __type196 __type197;
    typedef decltype(std::declval<__type197>()[std::declval<__type91>()]) __type198;
    typedef decltype(pythonic::operator_::mul(std::declval<__type198>(), std::declval<__type116>())) __type201;
    typedef container<typename std::remove_reference<__type201>::type> __type202;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195,__type202>::type __type203;
    typedef __type203 __type204;
    typedef decltype(std::declval<__type204>()[std::declval<__type91>()]) __type205;
    typedef decltype(pythonic::operator_::mul(std::declval<__type205>(), std::declval<__type123>())) __type208;
    typedef container<typename std::remove_reference<__type208>::type> __type209;
    typedef typename __combined<__type89,__type92,__type97,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195,__type202,__type209>::type __type210;
    typedef typename pythonic::assignable<__type210>::type __type211;
    typedef typename pythonic::assignable<__type43>::type __type212;
    typedef typename pythonic::assignable<__type79>::type __type213;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type50 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type86 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type211 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      long  __target140298636703056 = pythonic::builtins::functor::len{}(x);
      for (long  i=0L; i < __target140298636703056; i += 1L)
      {
        __type212 __tuple0 = cu_find_span()(xmin, dx, x[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span1 = std::get<0>(__tuple0);
        __type213 __tuple1 = cu_find_span()(ymin, dy, y[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span2 = std::get<0>(__tuple1);
        cu_basis_funs_1st_der()(span1, std::get<1>(__tuple0), dx, basis1);
        cu_basis_funs_1st_der()(span2, std::get<1>(__tuple1), dy, basis2);
        theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
        z.fast(i) = 0.0;
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_vector_00::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_vector_00::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
    typedef __type22 __type23;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type24;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type25;
    typedef decltype(std::declval<__type25>()(std::declval<__type23>())) __type27;
    typedef decltype(std::declval<__type24>()(std::declval<__type27>())) __type28;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type28>::type::iterator>::value_type>::type __type29;
    typedef __type29 __type30;
    typedef decltype(std::declval<__type23>()[std::declval<__type30>()]) __type31;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type31>())) __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
    typedef typename pythonic::assignable<__type35>::type __type36;
    typedef __type36 __type37;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type34>::type>::type __type38;
    typedef __type5 __type39;
    typedef typename cu_basis_funs::type<__type37, __type38, __type39>::__ptype21 __type40;
    typedef indexable_container<__type14, typename std::remove_reference<__type40>::type> __type41;
    typedef typename __combined<__type33,__type41>::type __type42;
    typedef typename cu_basis_funs::type<__type37, __type38, __type39>::__ptype22 __type43;
    typedef indexable_container<__type6, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type39,__type44>::type __type45;
    typedef typename cu_basis_funs::type<__type37, __type38, __type45>::__ptype23 __type46;
    typedef indexable_container<__type6, typename std::remove_reference<__type46>::type> __type47;
    typedef typename __combined<__type5,__type44,__type47,__type39>::type __type48;
    typedef typename pythonic::assignable<__type48>::type __type49;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type50;
    typedef __type50 __type51;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type51>::type>::type __type52;
    typedef typename pythonic::assignable<__type52>::type __type53;
    typedef __type53 __type54;
    typedef indexable_container<__type6, typename std::remove_reference<__type52>::type> __type55;
    typedef typename __combined<__type50,__type55>::type __type56;
    typedef __type56 __type57;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type57>::type>::type __type58;
    typedef indexable_container<__type14, typename std::remove_reference<__type58>::type> __type59;
    typedef typename __combined<__type50,__type55,__type59>::type __type60;
    typedef typename pythonic::assignable<__type58>::type __type61;
    typedef __type61 __type62;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type63;
    typedef __type63 __type64;
    typedef decltype(std::declval<__type64>()[std::declval<__type30>()]) __type66;
    typedef decltype(std::declval<__type7>()(std::declval<__type54>(), std::declval<__type62>(), std::declval<__type66>())) __type67;
    typedef typename pythonic::assignable<__type67>::type __type68;
    typedef __type68 __type69;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type69>::type>::type __type70;
    typedef typename pythonic::assignable<__type70>::type __type71;
    typedef __type71 __type72;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type69>::type>::type __type73;
    typedef typename cu_basis_funs::type<__type72, __type73, __type39>::__ptype21 __type75;
    typedef indexable_container<__type14, typename std::remove_reference<__type75>::type> __type76;
    typedef typename __combined<__type68,__type76>::type __type77;
    typedef typename cu_basis_funs::type<__type72, __type73, __type39>::__ptype22 __type78;
    typedef indexable_container<__type6, typename std::remove_reference<__type78>::type> __type79;
    typedef typename __combined<__type39,__type79>::type __type80;
    typedef typename cu_basis_funs::type<__type72, __type73, __type80>::__ptype23 __type81;
    typedef indexable_container<__type6, typename std::remove_reference<__type81>::type> __type82;
    typedef typename __combined<__type5,__type79,__type82,__type39>::type __type83;
    typedef typename pythonic::assignable<__type83>::type __type84;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type85;
    typedef decltype(std::declval<__type0>()(std::declval<__type85>(), std::declval<__type3>())) __type86;
    typedef typename pythonic::assignable<__type86>::type __type87;
    typedef long __type88;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type88>(), std::declval<__type88>())) __type89;
    typedef indexable<__type89> __type90;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type91;
    typedef __type91 __type92;
    typedef pythonic::types::contiguous_slice __type93;
    typedef decltype(std::declval<__type92>()(std::declval<__type93>(), std::declval<__type93>())) __type94;
    typedef container<typename std::remove_reference<__type94>::type> __type95;
    typedef typename __combined<__type87,__type95>::type __type96;
    typedef __type96 __type97;
    typedef decltype(std::declval<__type97>()[std::declval<__type89>()]) __type98;
    typedef __type83 __type99;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type99>::type>::type __type100;
    typedef decltype(pythonic::operator_::mul(std::declval<__type98>(), std::declval<__type100>())) __type101;
    typedef container<typename std::remove_reference<__type101>::type> __type102;
    typedef typename __combined<__type87,__type90,__type95,__type102>::type __type103;
    typedef __type103 __type104;
    typedef decltype(std::declval<__type104>()[std::declval<__type89>()]) __type105;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type99>::type>::type __type107;
    typedef decltype(pythonic::operator_::mul(std::declval<__type105>(), std::declval<__type107>())) __type108;
    typedef container<typename std::remove_reference<__type108>::type> __type109;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109>::type __type110;
    typedef __type110 __type111;
    typedef decltype(std::declval<__type111>()[std::declval<__type89>()]) __type112;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type99>::type>::type __type114;
    typedef decltype(pythonic::operator_::mul(std::declval<__type112>(), std::declval<__type114>())) __type115;
    typedef container<typename std::remove_reference<__type115>::type> __type116;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116>::type __type117;
    typedef __type117 __type118;
    typedef decltype(std::declval<__type118>()[std::declval<__type89>()]) __type119;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type99>::type>::type __type121;
    typedef decltype(pythonic::operator_::mul(std::declval<__type119>(), std::declval<__type121>())) __type122;
    typedef container<typename std::remove_reference<__type122>::type> __type123;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123>::type __type124;
    typedef __type124 __type125;
    typedef decltype(std::declval<__type125>()[std::declval<__type89>()]) __type126;
    typedef decltype(pythonic::operator_::mul(std::declval<__type126>(), std::declval<__type100>())) __type129;
    typedef container<typename std::remove_reference<__type129>::type> __type130;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130>::type __type131;
    typedef __type131 __type132;
    typedef decltype(std::declval<__type132>()[std::declval<__type89>()]) __type133;
    typedef decltype(pythonic::operator_::mul(std::declval<__type133>(), std::declval<__type107>())) __type136;
    typedef container<typename std::remove_reference<__type136>::type> __type137;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137>::type __type138;
    typedef __type138 __type139;
    typedef decltype(std::declval<__type139>()[std::declval<__type89>()]) __type140;
    typedef decltype(pythonic::operator_::mul(std::declval<__type140>(), std::declval<__type114>())) __type143;
    typedef container<typename std::remove_reference<__type143>::type> __type144;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144>::type __type145;
    typedef __type145 __type146;
    typedef decltype(std::declval<__type146>()[std::declval<__type89>()]) __type147;
    typedef decltype(pythonic::operator_::mul(std::declval<__type147>(), std::declval<__type121>())) __type150;
    typedef container<typename std::remove_reference<__type150>::type> __type151;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151>::type __type152;
    typedef __type152 __type153;
    typedef decltype(std::declval<__type153>()[std::declval<__type89>()]) __type154;
    typedef decltype(pythonic::operator_::mul(std::declval<__type154>(), std::declval<__type100>())) __type157;
    typedef container<typename std::remove_reference<__type157>::type> __type158;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158>::type __type159;
    typedef __type159 __type160;
    typedef decltype(std::declval<__type160>()[std::declval<__type89>()]) __type161;
    typedef decltype(pythonic::operator_::mul(std::declval<__type161>(), std::declval<__type107>())) __type164;
    typedef container<typename std::remove_reference<__type164>::type> __type165;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165>::type __type166;
    typedef __type166 __type167;
    typedef decltype(std::declval<__type167>()[std::declval<__type89>()]) __type168;
    typedef decltype(pythonic::operator_::mul(std::declval<__type168>(), std::declval<__type114>())) __type171;
    typedef container<typename std::remove_reference<__type171>::type> __type172;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172>::type __type173;
    typedef __type173 __type174;
    typedef decltype(std::declval<__type174>()[std::declval<__type89>()]) __type175;
    typedef decltype(pythonic::operator_::mul(std::declval<__type175>(), std::declval<__type121>())) __type178;
    typedef container<typename std::remove_reference<__type178>::type> __type179;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179>::type __type180;
    typedef __type180 __type181;
    typedef decltype(std::declval<__type181>()[std::declval<__type89>()]) __type182;
    typedef decltype(pythonic::operator_::mul(std::declval<__type182>(), std::declval<__type100>())) __type185;
    typedef container<typename std::remove_reference<__type185>::type> __type186;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186>::type __type187;
    typedef __type187 __type188;
    typedef decltype(std::declval<__type188>()[std::declval<__type89>()]) __type189;
    typedef decltype(pythonic::operator_::mul(std::declval<__type189>(), std::declval<__type107>())) __type192;
    typedef container<typename std::remove_reference<__type192>::type> __type193;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193>::type __type194;
    typedef __type194 __type195;
    typedef decltype(std::declval<__type195>()[std::declval<__type89>()]) __type196;
    typedef decltype(pythonic::operator_::mul(std::declval<__type196>(), std::declval<__type114>())) __type199;
    typedef container<typename std::remove_reference<__type199>::type> __type200;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200>::type __type201;
    typedef __type201 __type202;
    typedef decltype(std::declval<__type202>()[std::declval<__type89>()]) __type203;
    typedef decltype(pythonic::operator_::mul(std::declval<__type203>(), std::declval<__type121>())) __type206;
    typedef container<typename std::remove_reference<__type206>::type> __type207;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200,__type207>::type __type208;
    typedef typename pythonic::assignable<__type208>::type __type209;
    typedef typename pythonic::assignable<__type42>::type __type210;
    typedef typename pythonic::assignable<__type77>::type __type211;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type49 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type84 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type209 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      long  __target140298636485056 = pythonic::builtins::functor::len{}(x);
      for (long  i=0L; i < __target140298636485056; i += 1L)
      {
        __type210 __tuple0 = cu_find_span()(xmin, dx, x[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span1 = std::get<0>(__tuple0);
        __type211 __tuple1 = cu_find_span()(ymin, dy, y[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span2 = std::get<0>(__tuple1);
        cu_basis_funs()(span1, std::get<1>(__tuple0), basis1);
        cu_basis_funs()(span2, std::get<1>(__tuple1), basis2);
        theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
        z.fast(i) = 0.0;
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
  inline
  typename cu_eval_spline_2d_cross_11::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type cu_eval_spline_2d_cross_11::operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type22;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type23;
    typedef __type23 __type24;
    typedef decltype(std::declval<__type22>()(std::declval<__type24>())) __type25;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type25>::type::iterator>::value_type>::type __type26;
    typedef __type26 __type27;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type27>::type>::type __type28;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type28>())) __type29;
    typedef typename pythonic::assignable<__type29>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type35;
    typedef __type5 __type37;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type37>::__ptype0 __type38;
    typedef indexable_container<__type14, typename std::remove_reference<__type38>::type> __type39;
    typedef typename __combined<__type30,__type39>::type __type40;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type37>::__ptype1 __type41;
    typedef indexable_container<__type6, typename std::remove_reference<__type41>::type> __type42;
    typedef typename __combined<__type37,__type42>::type __type43;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type43>::__ptype2 __type44;
    typedef indexable_container<__type6, typename std::remove_reference<__type44>::type> __type45;
    typedef typename __combined<__type5,__type42,__type45,__type37>::type __type46;
    typedef typename pythonic::assignable<__type46>::type __type47;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type48;
    typedef __type48 __type49;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type49>::type>::type __type50;
    typedef typename pythonic::assignable<__type50>::type __type51;
    typedef __type51 __type52;
    typedef indexable_container<__type6, typename std::remove_reference<__type50>::type> __type53;
    typedef typename __combined<__type48,__type53>::type __type54;
    typedef __type54 __type55;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type55>::type>::type __type56;
    typedef indexable_container<__type14, typename std::remove_reference<__type56>::type> __type57;
    typedef typename __combined<__type48,__type53,__type57>::type __type58;
    typedef typename pythonic::assignable<__type56>::type __type59;
    typedef __type59 __type60;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type61;
    typedef __type61 __type62;
    typedef decltype(std::declval<__type22>()(std::declval<__type62>())) __type63;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type63>::type::iterator>::value_type>::type __type64;
    typedef __type64 __type65;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type65>::type>::type __type66;
    typedef decltype(std::declval<__type7>()(std::declval<__type52>(), std::declval<__type60>(), std::declval<__type66>())) __type67;
    typedef typename pythonic::assignable<__type67>::type __type68;
    typedef __type68 __type69;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type69>::type>::type __type70;
    typedef typename pythonic::assignable<__type70>::type __type71;
    typedef __type71 __type72;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type69>::type>::type __type73;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type21, __type37>::__ptype0 __type76;
    typedef indexable_container<__type14, typename std::remove_reference<__type76>::type> __type77;
    typedef typename __combined<__type68,__type77>::type __type78;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type21, __type37>::__ptype1 __type79;
    typedef indexable_container<__type6, typename std::remove_reference<__type79>::type> __type80;
    typedef typename __combined<__type37,__type80>::type __type81;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type21, __type81>::__ptype2 __type82;
    typedef indexable_container<__type6, typename std::remove_reference<__type82>::type> __type83;
    typedef typename __combined<__type5,__type80,__type83,__type37>::type __type84;
    typedef typename pythonic::assignable<__type84>::type __type85;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type86;
    typedef decltype(std::declval<__type0>()(std::declval<__type86>(), std::declval<__type3>())) __type87;
    typedef typename pythonic::assignable<__type87>::type __type88;
    typedef long __type89;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type89>(), std::declval<__type89>())) __type90;
    typedef indexable<__type90> __type91;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type92;
    typedef __type92 __type93;
    typedef pythonic::types::contiguous_slice __type94;
    typedef decltype(std::declval<__type93>()(std::declval<__type94>(), std::declval<__type94>())) __type95;
    typedef container<typename std::remove_reference<__type95>::type> __type96;
    typedef typename __combined<__type88,__type96>::type __type97;
    typedef __type97 __type98;
    typedef decltype(std::declval<__type98>()[std::declval<__type90>()]) __type99;
    typedef __type84 __type100;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type100>::type>::type __type101;
    typedef decltype(pythonic::operator_::mul(std::declval<__type99>(), std::declval<__type101>())) __type102;
    typedef container<typename std::remove_reference<__type102>::type> __type103;
    typedef typename __combined<__type88,__type91,__type96,__type103>::type __type104;
    typedef __type104 __type105;
    typedef decltype(std::declval<__type105>()[std::declval<__type90>()]) __type106;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type100>::type>::type __type108;
    typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type108>())) __type109;
    typedef container<typename std::remove_reference<__type109>::type> __type110;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110>::type __type111;
    typedef __type111 __type112;
    typedef decltype(std::declval<__type112>()[std::declval<__type90>()]) __type113;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type100>::type>::type __type115;
    typedef decltype(pythonic::operator_::mul(std::declval<__type113>(), std::declval<__type115>())) __type116;
    typedef container<typename std::remove_reference<__type116>::type> __type117;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117>::type __type118;
    typedef __type118 __type119;
    typedef decltype(std::declval<__type119>()[std::declval<__type90>()]) __type120;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type100>::type>::type __type122;
    typedef decltype(pythonic::operator_::mul(std::declval<__type120>(), std::declval<__type122>())) __type123;
    typedef container<typename std::remove_reference<__type123>::type> __type124;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124>::type __type125;
    typedef __type125 __type126;
    typedef decltype(std::declval<__type126>()[std::declval<__type90>()]) __type127;
    typedef decltype(pythonic::operator_::mul(std::declval<__type127>(), std::declval<__type101>())) __type130;
    typedef container<typename std::remove_reference<__type130>::type> __type131;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131>::type __type132;
    typedef __type132 __type133;
    typedef decltype(std::declval<__type133>()[std::declval<__type90>()]) __type134;
    typedef decltype(pythonic::operator_::mul(std::declval<__type134>(), std::declval<__type108>())) __type137;
    typedef container<typename std::remove_reference<__type137>::type> __type138;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138>::type __type139;
    typedef __type139 __type140;
    typedef decltype(std::declval<__type140>()[std::declval<__type90>()]) __type141;
    typedef decltype(pythonic::operator_::mul(std::declval<__type141>(), std::declval<__type115>())) __type144;
    typedef container<typename std::remove_reference<__type144>::type> __type145;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145>::type __type146;
    typedef __type146 __type147;
    typedef decltype(std::declval<__type147>()[std::declval<__type90>()]) __type148;
    typedef decltype(pythonic::operator_::mul(std::declval<__type148>(), std::declval<__type122>())) __type151;
    typedef container<typename std::remove_reference<__type151>::type> __type152;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152>::type __type153;
    typedef __type153 __type154;
    typedef decltype(std::declval<__type154>()[std::declval<__type90>()]) __type155;
    typedef decltype(pythonic::operator_::mul(std::declval<__type155>(), std::declval<__type101>())) __type158;
    typedef container<typename std::remove_reference<__type158>::type> __type159;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159>::type __type160;
    typedef __type160 __type161;
    typedef decltype(std::declval<__type161>()[std::declval<__type90>()]) __type162;
    typedef decltype(pythonic::operator_::mul(std::declval<__type162>(), std::declval<__type108>())) __type165;
    typedef container<typename std::remove_reference<__type165>::type> __type166;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166>::type __type167;
    typedef __type167 __type168;
    typedef decltype(std::declval<__type168>()[std::declval<__type90>()]) __type169;
    typedef decltype(pythonic::operator_::mul(std::declval<__type169>(), std::declval<__type115>())) __type172;
    typedef container<typename std::remove_reference<__type172>::type> __type173;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173>::type __type174;
    typedef __type174 __type175;
    typedef decltype(std::declval<__type175>()[std::declval<__type90>()]) __type176;
    typedef decltype(pythonic::operator_::mul(std::declval<__type176>(), std::declval<__type122>())) __type179;
    typedef container<typename std::remove_reference<__type179>::type> __type180;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180>::type __type181;
    typedef __type181 __type182;
    typedef decltype(std::declval<__type182>()[std::declval<__type90>()]) __type183;
    typedef decltype(pythonic::operator_::mul(std::declval<__type183>(), std::declval<__type101>())) __type186;
    typedef container<typename std::remove_reference<__type186>::type> __type187;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187>::type __type188;
    typedef __type188 __type189;
    typedef decltype(std::declval<__type189>()[std::declval<__type90>()]) __type190;
    typedef decltype(pythonic::operator_::mul(std::declval<__type190>(), std::declval<__type108>())) __type193;
    typedef container<typename std::remove_reference<__type193>::type> __type194;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194>::type __type195;
    typedef __type195 __type196;
    typedef decltype(std::declval<__type196>()[std::declval<__type90>()]) __type197;
    typedef decltype(pythonic::operator_::mul(std::declval<__type197>(), std::declval<__type115>())) __type200;
    typedef container<typename std::remove_reference<__type200>::type> __type201;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201>::type __type202;
    typedef __type202 __type203;
    typedef decltype(std::declval<__type203>()[std::declval<__type90>()]) __type204;
    typedef decltype(pythonic::operator_::mul(std::declval<__type204>(), std::declval<__type122>())) __type207;
    typedef container<typename std::remove_reference<__type207>::type> __type208;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201,__type208>::type __type209;
    typedef typename pythonic::assignable<__type209>::type __type210;
    typedef typename pythonic::assignable<__type40>::type __type211;
    typedef typename pythonic::assignable<__type78>::type __type212;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type47 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type85 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type210 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      for (auto&& __tuple0: pythonic::builtins::functor::enumerate{}(X))
      {
        typename pythonic::lazy<decltype(std::get<0>(__tuple0))>::type i = std::get<0>(__tuple0);
        __type211 __tuple1 = cu_find_span()(xmin, dx, std::get<1>(__tuple0));
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span1 = std::get<0>(__tuple1);
        cu_basis_funs_1st_der()(span1, std::get<1>(__tuple1), dx, basis1);
        {
          for (auto&& __tuple2: pythonic::builtins::functor::enumerate{}(Y))
          {
            typename pythonic::lazy<decltype(std::get<0>(__tuple2))>::type j = std::get<0>(__tuple2);
            __type212 __tuple3 = cu_find_span()(ymin, dy, std::get<1>(__tuple2));
            typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple3))>::type span2 = std::get<0>(__tuple3);
            cu_basis_funs_1st_der()(span2, std::get<1>(__tuple3), dx, basis2);
            theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
            z[pythonic::types::make_tuple(i, j)] = 0.0;
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
  inline
  typename cu_eval_spline_2d_cross_00::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type cu_eval_spline_2d_cross_00::operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type22;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type23;
    typedef __type23 __type24;
    typedef decltype(std::declval<__type22>()(std::declval<__type24>())) __type25;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type25>::type::iterator>::value_type>::type __type26;
    typedef __type26 __type27;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type27>::type>::type __type28;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type28>())) __type29;
    typedef typename pythonic::assignable<__type29>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type35;
    typedef __type5 __type36;
    typedef typename cu_basis_funs::type<__type34, __type35, __type36>::__ptype21 __type37;
    typedef indexable_container<__type14, typename std::remove_reference<__type37>::type> __type38;
    typedef typename __combined<__type30,__type38>::type __type39;
    typedef typename cu_basis_funs::type<__type34, __type35, __type36>::__ptype22 __type40;
    typedef indexable_container<__type6, typename std::remove_reference<__type40>::type> __type41;
    typedef typename __combined<__type36,__type41>::type __type42;
    typedef typename cu_basis_funs::type<__type34, __type35, __type42>::__ptype23 __type43;
    typedef indexable_container<__type6, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type5,__type41,__type44,__type36>::type __type45;
    typedef typename pythonic::assignable<__type45>::type __type46;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type47;
    typedef __type47 __type48;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type48>::type>::type __type49;
    typedef typename pythonic::assignable<__type49>::type __type50;
    typedef __type50 __type51;
    typedef indexable_container<__type6, typename std::remove_reference<__type49>::type> __type52;
    typedef typename __combined<__type47,__type52>::type __type53;
    typedef __type53 __type54;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type54>::type>::type __type55;
    typedef indexable_container<__type14, typename std::remove_reference<__type55>::type> __type56;
    typedef typename __combined<__type47,__type52,__type56>::type __type57;
    typedef typename pythonic::assignable<__type55>::type __type58;
    typedef __type58 __type59;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type60;
    typedef __type60 __type61;
    typedef decltype(std::declval<__type22>()(std::declval<__type61>())) __type62;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type62>::type::iterator>::value_type>::type __type63;
    typedef __type63 __type64;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type64>::type>::type __type65;
    typedef decltype(std::declval<__type7>()(std::declval<__type51>(), std::declval<__type59>(), std::declval<__type65>())) __type66;
    typedef typename pythonic::assignable<__type66>::type __type67;
    typedef __type67 __type68;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type68>::type>::type __type69;
    typedef typename pythonic::assignable<__type69>::type __type70;
    typedef __type70 __type71;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type68>::type>::type __type72;
    typedef typename cu_basis_funs::type<__type71, __type72, __type36>::__ptype21 __type74;
    typedef indexable_container<__type14, typename std::remove_reference<__type74>::type> __type75;
    typedef typename __combined<__type67,__type75>::type __type76;
    typedef typename cu_basis_funs::type<__type71, __type72, __type36>::__ptype22 __type77;
    typedef indexable_container<__type6, typename std::remove_reference<__type77>::type> __type78;
    typedef typename __combined<__type36,__type78>::type __type79;
    typedef typename cu_basis_funs::type<__type71, __type72, __type79>::__ptype23 __type80;
    typedef indexable_container<__type6, typename std::remove_reference<__type80>::type> __type81;
    typedef typename __combined<__type5,__type78,__type81,__type36>::type __type82;
    typedef typename pythonic::assignable<__type82>::type __type83;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type84;
    typedef decltype(std::declval<__type0>()(std::declval<__type84>(), std::declval<__type3>())) __type85;
    typedef typename pythonic::assignable<__type85>::type __type86;
    typedef long __type87;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type87>(), std::declval<__type87>())) __type88;
    typedef indexable<__type88> __type89;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type90;
    typedef __type90 __type91;
    typedef pythonic::types::contiguous_slice __type92;
    typedef decltype(std::declval<__type91>()(std::declval<__type92>(), std::declval<__type92>())) __type93;
    typedef container<typename std::remove_reference<__type93>::type> __type94;
    typedef typename __combined<__type86,__type94>::type __type95;
    typedef __type95 __type96;
    typedef decltype(std::declval<__type96>()[std::declval<__type88>()]) __type97;
    typedef __type82 __type98;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type98>::type>::type __type99;
    typedef decltype(pythonic::operator_::mul(std::declval<__type97>(), std::declval<__type99>())) __type100;
    typedef container<typename std::remove_reference<__type100>::type> __type101;
    typedef typename __combined<__type86,__type89,__type94,__type101>::type __type102;
    typedef __type102 __type103;
    typedef decltype(std::declval<__type103>()[std::declval<__type88>()]) __type104;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type98>::type>::type __type106;
    typedef decltype(pythonic::operator_::mul(std::declval<__type104>(), std::declval<__type106>())) __type107;
    typedef container<typename std::remove_reference<__type107>::type> __type108;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108>::type __type109;
    typedef __type109 __type110;
    typedef decltype(std::declval<__type110>()[std::declval<__type88>()]) __type111;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type98>::type>::type __type113;
    typedef decltype(pythonic::operator_::mul(std::declval<__type111>(), std::declval<__type113>())) __type114;
    typedef container<typename std::remove_reference<__type114>::type> __type115;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115>::type __type116;
    typedef __type116 __type117;
    typedef decltype(std::declval<__type117>()[std::declval<__type88>()]) __type118;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type98>::type>::type __type120;
    typedef decltype(pythonic::operator_::mul(std::declval<__type118>(), std::declval<__type120>())) __type121;
    typedef container<typename std::remove_reference<__type121>::type> __type122;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122>::type __type123;
    typedef __type123 __type124;
    typedef decltype(std::declval<__type124>()[std::declval<__type88>()]) __type125;
    typedef decltype(pythonic::operator_::mul(std::declval<__type125>(), std::declval<__type99>())) __type128;
    typedef container<typename std::remove_reference<__type128>::type> __type129;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129>::type __type130;
    typedef __type130 __type131;
    typedef decltype(std::declval<__type131>()[std::declval<__type88>()]) __type132;
    typedef decltype(pythonic::operator_::mul(std::declval<__type132>(), std::declval<__type106>())) __type135;
    typedef container<typename std::remove_reference<__type135>::type> __type136;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136>::type __type137;
    typedef __type137 __type138;
    typedef decltype(std::declval<__type138>()[std::declval<__type88>()]) __type139;
    typedef decltype(pythonic::operator_::mul(std::declval<__type139>(), std::declval<__type113>())) __type142;
    typedef container<typename std::remove_reference<__type142>::type> __type143;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143>::type __type144;
    typedef __type144 __type145;
    typedef decltype(std::declval<__type145>()[std::declval<__type88>()]) __type146;
    typedef decltype(pythonic::operator_::mul(std::declval<__type146>(), std::declval<__type120>())) __type149;
    typedef container<typename std::remove_reference<__type149>::type> __type150;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150>::type __type151;
    typedef __type151 __type152;
    typedef decltype(std::declval<__type152>()[std::declval<__type88>()]) __type153;
    typedef decltype(pythonic::operator_::mul(std::declval<__type153>(), std::declval<__type99>())) __type156;
    typedef container<typename std::remove_reference<__type156>::type> __type157;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157>::type __type158;
    typedef __type158 __type159;
    typedef decltype(std::declval<__type159>()[std::declval<__type88>()]) __type160;
    typedef decltype(pythonic::operator_::mul(std::declval<__type160>(), std::declval<__type106>())) __type163;
    typedef container<typename std::remove_reference<__type163>::type> __type164;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164>::type __type165;
    typedef __type165 __type166;
    typedef decltype(std::declval<__type166>()[std::declval<__type88>()]) __type167;
    typedef decltype(pythonic::operator_::mul(std::declval<__type167>(), std::declval<__type113>())) __type170;
    typedef container<typename std::remove_reference<__type170>::type> __type171;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171>::type __type172;
    typedef __type172 __type173;
    typedef decltype(std::declval<__type173>()[std::declval<__type88>()]) __type174;
    typedef decltype(pythonic::operator_::mul(std::declval<__type174>(), std::declval<__type120>())) __type177;
    typedef container<typename std::remove_reference<__type177>::type> __type178;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171,__type178>::type __type179;
    typedef __type179 __type180;
    typedef decltype(std::declval<__type180>()[std::declval<__type88>()]) __type181;
    typedef decltype(pythonic::operator_::mul(std::declval<__type181>(), std::declval<__type99>())) __type184;
    typedef container<typename std::remove_reference<__type184>::type> __type185;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171,__type178,__type185>::type __type186;
    typedef __type186 __type187;
    typedef decltype(std::declval<__type187>()[std::declval<__type88>()]) __type188;
    typedef decltype(pythonic::operator_::mul(std::declval<__type188>(), std::declval<__type106>())) __type191;
    typedef container<typename std::remove_reference<__type191>::type> __type192;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171,__type178,__type185,__type192>::type __type193;
    typedef __type193 __type194;
    typedef decltype(std::declval<__type194>()[std::declval<__type88>()]) __type195;
    typedef decltype(pythonic::operator_::mul(std::declval<__type195>(), std::declval<__type113>())) __type198;
    typedef container<typename std::remove_reference<__type198>::type> __type199;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171,__type178,__type185,__type192,__type199>::type __type200;
    typedef __type200 __type201;
    typedef decltype(std::declval<__type201>()[std::declval<__type88>()]) __type202;
    typedef decltype(pythonic::operator_::mul(std::declval<__type202>(), std::declval<__type120>())) __type205;
    typedef container<typename std::remove_reference<__type205>::type> __type206;
    typedef typename __combined<__type86,__type89,__type94,__type101,__type108,__type115,__type122,__type129,__type136,__type143,__type150,__type157,__type164,__type171,__type178,__type185,__type192,__type199,__type206>::type __type207;
    typedef typename pythonic::assignable<__type207>::type __type208;
    typedef typename pythonic::assignable<__type39>::type __type209;
    typedef typename pythonic::assignable<__type76>::type __type210;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type46 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type83 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type208 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      for (auto&& __tuple0: pythonic::builtins::functor::enumerate{}(X))
      {
        typename pythonic::lazy<decltype(std::get<0>(__tuple0))>::type i = std::get<0>(__tuple0);
        __type209 __tuple1 = cu_find_span()(xmin, dx, std::get<1>(__tuple0));
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span1 = std::get<0>(__tuple1);
        cu_basis_funs()(span1, std::get<1>(__tuple1), basis1);
        {
          for (auto&& __tuple2: pythonic::builtins::functor::enumerate{}(Y))
          {
            typename pythonic::lazy<decltype(std::get<0>(__tuple2))>::type j = std::get<0>(__tuple2);
            __type210 __tuple3 = cu_find_span()(ymin, dy, std::get<1>(__tuple2));
            typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple3))>::type span2 = std::get<0>(__tuple3);
            cu_basis_funs()(span2, std::get<1>(__tuple3), basis2);
            theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
            z[pythonic::types::make_tuple(i, j)] = 0.0;
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_vector_10::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_vector_10::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
    typedef __type22 __type23;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type24;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type25;
    typedef decltype(std::declval<__type25>()(std::declval<__type23>())) __type27;
    typedef decltype(std::declval<__type24>()(std::declval<__type27>())) __type28;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type28>::type::iterator>::value_type>::type __type29;
    typedef __type29 __type30;
    typedef decltype(std::declval<__type23>()[std::declval<__type30>()]) __type31;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type31>())) __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
    typedef typename pythonic::assignable<__type35>::type __type36;
    typedef __type36 __type37;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type34>::type>::type __type38;
    typedef __type5 __type40;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type40>::__ptype0 __type41;
    typedef indexable_container<__type14, typename std::remove_reference<__type41>::type> __type42;
    typedef typename __combined<__type33,__type42>::type __type43;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type40>::__ptype1 __type44;
    typedef indexable_container<__type6, typename std::remove_reference<__type44>::type> __type45;
    typedef typename __combined<__type40,__type45>::type __type46;
    typedef typename cu_basis_funs_1st_der::type<__type37, __type38, __type21, __type46>::__ptype2 __type47;
    typedef indexable_container<__type6, typename std::remove_reference<__type47>::type> __type48;
    typedef typename __combined<__type5,__type45,__type48,__type40>::type __type49;
    typedef typename pythonic::assignable<__type49>::type __type50;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type51;
    typedef __type51 __type52;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type52>::type>::type __type53;
    typedef typename pythonic::assignable<__type53>::type __type54;
    typedef __type54 __type55;
    typedef indexable_container<__type6, typename std::remove_reference<__type53>::type> __type56;
    typedef typename __combined<__type51,__type56>::type __type57;
    typedef __type57 __type58;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type58>::type>::type __type59;
    typedef indexable_container<__type14, typename std::remove_reference<__type59>::type> __type60;
    typedef typename __combined<__type51,__type56,__type60>::type __type61;
    typedef typename pythonic::assignable<__type59>::type __type62;
    typedef __type62 __type63;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type64;
    typedef __type64 __type65;
    typedef decltype(std::declval<__type65>()[std::declval<__type30>()]) __type67;
    typedef decltype(std::declval<__type7>()(std::declval<__type55>(), std::declval<__type63>(), std::declval<__type67>())) __type68;
    typedef typename pythonic::assignable<__type68>::type __type69;
    typedef __type69 __type70;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type70>::type>::type __type71;
    typedef typename pythonic::assignable<__type71>::type __type72;
    typedef __type72 __type73;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type70>::type>::type __type74;
    typedef typename cu_basis_funs::type<__type73, __type74, __type40>::__ptype21 __type76;
    typedef indexable_container<__type14, typename std::remove_reference<__type76>::type> __type77;
    typedef typename __combined<__type69,__type77>::type __type78;
    typedef typename cu_basis_funs::type<__type73, __type74, __type40>::__ptype22 __type79;
    typedef indexable_container<__type6, typename std::remove_reference<__type79>::type> __type80;
    typedef typename __combined<__type40,__type80>::type __type81;
    typedef typename cu_basis_funs::type<__type73, __type74, __type81>::__ptype23 __type82;
    typedef indexable_container<__type6, typename std::remove_reference<__type82>::type> __type83;
    typedef typename __combined<__type5,__type80,__type83,__type40>::type __type84;
    typedef typename pythonic::assignable<__type84>::type __type85;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type86;
    typedef decltype(std::declval<__type0>()(std::declval<__type86>(), std::declval<__type3>())) __type87;
    typedef typename pythonic::assignable<__type87>::type __type88;
    typedef long __type89;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type89>(), std::declval<__type89>())) __type90;
    typedef indexable<__type90> __type91;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type92;
    typedef __type92 __type93;
    typedef pythonic::types::contiguous_slice __type94;
    typedef decltype(std::declval<__type93>()(std::declval<__type94>(), std::declval<__type94>())) __type95;
    typedef container<typename std::remove_reference<__type95>::type> __type96;
    typedef typename __combined<__type88,__type96>::type __type97;
    typedef __type97 __type98;
    typedef decltype(std::declval<__type98>()[std::declval<__type90>()]) __type99;
    typedef __type84 __type100;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type100>::type>::type __type101;
    typedef decltype(pythonic::operator_::mul(std::declval<__type99>(), std::declval<__type101>())) __type102;
    typedef container<typename std::remove_reference<__type102>::type> __type103;
    typedef typename __combined<__type88,__type91,__type96,__type103>::type __type104;
    typedef __type104 __type105;
    typedef decltype(std::declval<__type105>()[std::declval<__type90>()]) __type106;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type100>::type>::type __type108;
    typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type108>())) __type109;
    typedef container<typename std::remove_reference<__type109>::type> __type110;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110>::type __type111;
    typedef __type111 __type112;
    typedef decltype(std::declval<__type112>()[std::declval<__type90>()]) __type113;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type100>::type>::type __type115;
    typedef decltype(pythonic::operator_::mul(std::declval<__type113>(), std::declval<__type115>())) __type116;
    typedef container<typename std::remove_reference<__type116>::type> __type117;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117>::type __type118;
    typedef __type118 __type119;
    typedef decltype(std::declval<__type119>()[std::declval<__type90>()]) __type120;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type100>::type>::type __type122;
    typedef decltype(pythonic::operator_::mul(std::declval<__type120>(), std::declval<__type122>())) __type123;
    typedef container<typename std::remove_reference<__type123>::type> __type124;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124>::type __type125;
    typedef __type125 __type126;
    typedef decltype(std::declval<__type126>()[std::declval<__type90>()]) __type127;
    typedef decltype(pythonic::operator_::mul(std::declval<__type127>(), std::declval<__type101>())) __type130;
    typedef container<typename std::remove_reference<__type130>::type> __type131;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131>::type __type132;
    typedef __type132 __type133;
    typedef decltype(std::declval<__type133>()[std::declval<__type90>()]) __type134;
    typedef decltype(pythonic::operator_::mul(std::declval<__type134>(), std::declval<__type108>())) __type137;
    typedef container<typename std::remove_reference<__type137>::type> __type138;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138>::type __type139;
    typedef __type139 __type140;
    typedef decltype(std::declval<__type140>()[std::declval<__type90>()]) __type141;
    typedef decltype(pythonic::operator_::mul(std::declval<__type141>(), std::declval<__type115>())) __type144;
    typedef container<typename std::remove_reference<__type144>::type> __type145;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145>::type __type146;
    typedef __type146 __type147;
    typedef decltype(std::declval<__type147>()[std::declval<__type90>()]) __type148;
    typedef decltype(pythonic::operator_::mul(std::declval<__type148>(), std::declval<__type122>())) __type151;
    typedef container<typename std::remove_reference<__type151>::type> __type152;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152>::type __type153;
    typedef __type153 __type154;
    typedef decltype(std::declval<__type154>()[std::declval<__type90>()]) __type155;
    typedef decltype(pythonic::operator_::mul(std::declval<__type155>(), std::declval<__type101>())) __type158;
    typedef container<typename std::remove_reference<__type158>::type> __type159;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159>::type __type160;
    typedef __type160 __type161;
    typedef decltype(std::declval<__type161>()[std::declval<__type90>()]) __type162;
    typedef decltype(pythonic::operator_::mul(std::declval<__type162>(), std::declval<__type108>())) __type165;
    typedef container<typename std::remove_reference<__type165>::type> __type166;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166>::type __type167;
    typedef __type167 __type168;
    typedef decltype(std::declval<__type168>()[std::declval<__type90>()]) __type169;
    typedef decltype(pythonic::operator_::mul(std::declval<__type169>(), std::declval<__type115>())) __type172;
    typedef container<typename std::remove_reference<__type172>::type> __type173;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173>::type __type174;
    typedef __type174 __type175;
    typedef decltype(std::declval<__type175>()[std::declval<__type90>()]) __type176;
    typedef decltype(pythonic::operator_::mul(std::declval<__type176>(), std::declval<__type122>())) __type179;
    typedef container<typename std::remove_reference<__type179>::type> __type180;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180>::type __type181;
    typedef __type181 __type182;
    typedef decltype(std::declval<__type182>()[std::declval<__type90>()]) __type183;
    typedef decltype(pythonic::operator_::mul(std::declval<__type183>(), std::declval<__type101>())) __type186;
    typedef container<typename std::remove_reference<__type186>::type> __type187;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187>::type __type188;
    typedef __type188 __type189;
    typedef decltype(std::declval<__type189>()[std::declval<__type90>()]) __type190;
    typedef decltype(pythonic::operator_::mul(std::declval<__type190>(), std::declval<__type108>())) __type193;
    typedef container<typename std::remove_reference<__type193>::type> __type194;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194>::type __type195;
    typedef __type195 __type196;
    typedef decltype(std::declval<__type196>()[std::declval<__type90>()]) __type197;
    typedef decltype(pythonic::operator_::mul(std::declval<__type197>(), std::declval<__type115>())) __type200;
    typedef container<typename std::remove_reference<__type200>::type> __type201;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201>::type __type202;
    typedef __type202 __type203;
    typedef decltype(std::declval<__type203>()[std::declval<__type90>()]) __type204;
    typedef decltype(pythonic::operator_::mul(std::declval<__type204>(), std::declval<__type122>())) __type207;
    typedef container<typename std::remove_reference<__type207>::type> __type208;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201,__type208>::type __type209;
    typedef typename pythonic::assignable<__type209>::type __type210;
    typedef typename pythonic::assignable<__type43>::type __type211;
    typedef typename pythonic::assignable<__type78>::type __type212;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type50 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type85 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type210 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      long  __target140298636611200 = pythonic::builtins::functor::len{}(x);
      for (long  i=0L; i < __target140298636611200; i += 1L)
      {
        __type211 __tuple0 = cu_find_span()(xmin, dx, x[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span1 = std::get<0>(__tuple0);
        __type212 __tuple1 = cu_find_span()(ymin, dy, y[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span2 = std::get<0>(__tuple1);
        cu_basis_funs_1st_der()(span1, std::get<1>(__tuple0), dx, basis1);
        cu_basis_funs()(span2, std::get<1>(__tuple1), basis2);
        theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
        z.fast(i) = 0.0;
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_vector_01::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_vector_01::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
    typedef __type22 __type23;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::range{})>::type>::type __type24;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::len{})>::type>::type __type25;
    typedef decltype(std::declval<__type25>()(std::declval<__type23>())) __type27;
    typedef decltype(std::declval<__type24>()(std::declval<__type27>())) __type28;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type28>::type::iterator>::value_type>::type __type29;
    typedef __type29 __type30;
    typedef decltype(std::declval<__type23>()[std::declval<__type30>()]) __type31;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type31>())) __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type34>::type>::type __type35;
    typedef typename pythonic::assignable<__type35>::type __type36;
    typedef __type36 __type37;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type34>::type>::type __type38;
    typedef __type5 __type39;
    typedef typename cu_basis_funs::type<__type37, __type38, __type39>::__ptype21 __type40;
    typedef indexable_container<__type14, typename std::remove_reference<__type40>::type> __type41;
    typedef typename __combined<__type33,__type41>::type __type42;
    typedef typename cu_basis_funs::type<__type37, __type38, __type39>::__ptype22 __type43;
    typedef indexable_container<__type6, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type39,__type44>::type __type45;
    typedef typename cu_basis_funs::type<__type37, __type38, __type45>::__ptype23 __type46;
    typedef indexable_container<__type6, typename std::remove_reference<__type46>::type> __type47;
    typedef typename __combined<__type5,__type44,__type47,__type39>::type __type48;
    typedef typename pythonic::assignable<__type48>::type __type49;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type50;
    typedef __type50 __type51;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type51>::type>::type __type52;
    typedef typename pythonic::assignable<__type52>::type __type53;
    typedef __type53 __type54;
    typedef indexable_container<__type6, typename std::remove_reference<__type52>::type> __type55;
    typedef typename __combined<__type50,__type55>::type __type56;
    typedef __type56 __type57;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type57>::type>::type __type58;
    typedef indexable_container<__type14, typename std::remove_reference<__type58>::type> __type59;
    typedef typename __combined<__type50,__type55,__type59>::type __type60;
    typedef typename pythonic::assignable<__type58>::type __type61;
    typedef __type61 __type62;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type63;
    typedef __type63 __type64;
    typedef decltype(std::declval<__type64>()[std::declval<__type30>()]) __type66;
    typedef decltype(std::declval<__type7>()(std::declval<__type54>(), std::declval<__type62>(), std::declval<__type66>())) __type67;
    typedef typename pythonic::assignable<__type67>::type __type68;
    typedef __type68 __type69;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type69>::type>::type __type70;
    typedef typename pythonic::assignable<__type70>::type __type71;
    typedef __type71 __type72;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type69>::type>::type __type73;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type62, __type39>::__ptype0 __type76;
    typedef indexable_container<__type14, typename std::remove_reference<__type76>::type> __type77;
    typedef typename __combined<__type68,__type77>::type __type78;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type62, __type39>::__ptype1 __type79;
    typedef indexable_container<__type6, typename std::remove_reference<__type79>::type> __type80;
    typedef typename __combined<__type39,__type80>::type __type81;
    typedef typename cu_basis_funs_1st_der::type<__type72, __type73, __type62, __type81>::__ptype2 __type82;
    typedef indexable_container<__type6, typename std::remove_reference<__type82>::type> __type83;
    typedef typename __combined<__type5,__type80,__type83,__type39>::type __type84;
    typedef typename pythonic::assignable<__type84>::type __type85;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type86;
    typedef decltype(std::declval<__type0>()(std::declval<__type86>(), std::declval<__type3>())) __type87;
    typedef typename pythonic::assignable<__type87>::type __type88;
    typedef long __type89;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type89>(), std::declval<__type89>())) __type90;
    typedef indexable<__type90> __type91;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type92;
    typedef __type92 __type93;
    typedef pythonic::types::contiguous_slice __type94;
    typedef decltype(std::declval<__type93>()(std::declval<__type94>(), std::declval<__type94>())) __type95;
    typedef container<typename std::remove_reference<__type95>::type> __type96;
    typedef typename __combined<__type88,__type96>::type __type97;
    typedef __type97 __type98;
    typedef decltype(std::declval<__type98>()[std::declval<__type90>()]) __type99;
    typedef __type84 __type100;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type100>::type>::type __type101;
    typedef decltype(pythonic::operator_::mul(std::declval<__type99>(), std::declval<__type101>())) __type102;
    typedef container<typename std::remove_reference<__type102>::type> __type103;
    typedef typename __combined<__type88,__type91,__type96,__type103>::type __type104;
    typedef __type104 __type105;
    typedef decltype(std::declval<__type105>()[std::declval<__type90>()]) __type106;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type100>::type>::type __type108;
    typedef decltype(pythonic::operator_::mul(std::declval<__type106>(), std::declval<__type108>())) __type109;
    typedef container<typename std::remove_reference<__type109>::type> __type110;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110>::type __type111;
    typedef __type111 __type112;
    typedef decltype(std::declval<__type112>()[std::declval<__type90>()]) __type113;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type100>::type>::type __type115;
    typedef decltype(pythonic::operator_::mul(std::declval<__type113>(), std::declval<__type115>())) __type116;
    typedef container<typename std::remove_reference<__type116>::type> __type117;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117>::type __type118;
    typedef __type118 __type119;
    typedef decltype(std::declval<__type119>()[std::declval<__type90>()]) __type120;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type100>::type>::type __type122;
    typedef decltype(pythonic::operator_::mul(std::declval<__type120>(), std::declval<__type122>())) __type123;
    typedef container<typename std::remove_reference<__type123>::type> __type124;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124>::type __type125;
    typedef __type125 __type126;
    typedef decltype(std::declval<__type126>()[std::declval<__type90>()]) __type127;
    typedef decltype(pythonic::operator_::mul(std::declval<__type127>(), std::declval<__type101>())) __type130;
    typedef container<typename std::remove_reference<__type130>::type> __type131;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131>::type __type132;
    typedef __type132 __type133;
    typedef decltype(std::declval<__type133>()[std::declval<__type90>()]) __type134;
    typedef decltype(pythonic::operator_::mul(std::declval<__type134>(), std::declval<__type108>())) __type137;
    typedef container<typename std::remove_reference<__type137>::type> __type138;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138>::type __type139;
    typedef __type139 __type140;
    typedef decltype(std::declval<__type140>()[std::declval<__type90>()]) __type141;
    typedef decltype(pythonic::operator_::mul(std::declval<__type141>(), std::declval<__type115>())) __type144;
    typedef container<typename std::remove_reference<__type144>::type> __type145;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145>::type __type146;
    typedef __type146 __type147;
    typedef decltype(std::declval<__type147>()[std::declval<__type90>()]) __type148;
    typedef decltype(pythonic::operator_::mul(std::declval<__type148>(), std::declval<__type122>())) __type151;
    typedef container<typename std::remove_reference<__type151>::type> __type152;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152>::type __type153;
    typedef __type153 __type154;
    typedef decltype(std::declval<__type154>()[std::declval<__type90>()]) __type155;
    typedef decltype(pythonic::operator_::mul(std::declval<__type155>(), std::declval<__type101>())) __type158;
    typedef container<typename std::remove_reference<__type158>::type> __type159;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159>::type __type160;
    typedef __type160 __type161;
    typedef decltype(std::declval<__type161>()[std::declval<__type90>()]) __type162;
    typedef decltype(pythonic::operator_::mul(std::declval<__type162>(), std::declval<__type108>())) __type165;
    typedef container<typename std::remove_reference<__type165>::type> __type166;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166>::type __type167;
    typedef __type167 __type168;
    typedef decltype(std::declval<__type168>()[std::declval<__type90>()]) __type169;
    typedef decltype(pythonic::operator_::mul(std::declval<__type169>(), std::declval<__type115>())) __type172;
    typedef container<typename std::remove_reference<__type172>::type> __type173;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173>::type __type174;
    typedef __type174 __type175;
    typedef decltype(std::declval<__type175>()[std::declval<__type90>()]) __type176;
    typedef decltype(pythonic::operator_::mul(std::declval<__type176>(), std::declval<__type122>())) __type179;
    typedef container<typename std::remove_reference<__type179>::type> __type180;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180>::type __type181;
    typedef __type181 __type182;
    typedef decltype(std::declval<__type182>()[std::declval<__type90>()]) __type183;
    typedef decltype(pythonic::operator_::mul(std::declval<__type183>(), std::declval<__type101>())) __type186;
    typedef container<typename std::remove_reference<__type186>::type> __type187;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187>::type __type188;
    typedef __type188 __type189;
    typedef decltype(std::declval<__type189>()[std::declval<__type90>()]) __type190;
    typedef decltype(pythonic::operator_::mul(std::declval<__type190>(), std::declval<__type108>())) __type193;
    typedef container<typename std::remove_reference<__type193>::type> __type194;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194>::type __type195;
    typedef __type195 __type196;
    typedef decltype(std::declval<__type196>()[std::declval<__type90>()]) __type197;
    typedef decltype(pythonic::operator_::mul(std::declval<__type197>(), std::declval<__type115>())) __type200;
    typedef container<typename std::remove_reference<__type200>::type> __type201;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201>::type __type202;
    typedef __type202 __type203;
    typedef decltype(std::declval<__type203>()[std::declval<__type90>()]) __type204;
    typedef decltype(pythonic::operator_::mul(std::declval<__type204>(), std::declval<__type122>())) __type207;
    typedef container<typename std::remove_reference<__type207>::type> __type208;
    typedef typename __combined<__type88,__type91,__type96,__type103,__type110,__type117,__type124,__type131,__type138,__type145,__type152,__type159,__type166,__type173,__type180,__type187,__type194,__type201,__type208>::type __type209;
    typedef typename pythonic::assignable<__type209>::type __type210;
    typedef typename pythonic::assignable<__type42>::type __type211;
    typedef typename pythonic::assignable<__type78>::type __type212;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type49 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type85 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type210 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      long  __target140298636552176 = pythonic::builtins::functor::len{}(x);
      for (long  i=0L; i < __target140298636552176; i += 1L)
      {
        __type211 __tuple0 = cu_find_span()(xmin, dx, x[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span1 = std::get<0>(__tuple0);
        __type212 __tuple1 = cu_find_span()(ymin, dy, y[i]);
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span2 = std::get<0>(__tuple1);
        cu_basis_funs()(span1, std::get<1>(__tuple0), basis1);
        cu_basis_funs_1st_der()(span2, std::get<1>(__tuple1), dy, basis2);
        theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
        z.fast(i) = 0.0;
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
        theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
        z.fast(i) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
  inline
  typename cu_eval_spline_2d_cross_10::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type cu_eval_spline_2d_cross_10::operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type22;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type23;
    typedef __type23 __type24;
    typedef decltype(std::declval<__type22>()(std::declval<__type24>())) __type25;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type25>::type::iterator>::value_type>::type __type26;
    typedef __type26 __type27;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type27>::type>::type __type28;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type28>())) __type29;
    typedef typename pythonic::assignable<__type29>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type35;
    typedef __type5 __type37;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type37>::__ptype0 __type38;
    typedef indexable_container<__type14, typename std::remove_reference<__type38>::type> __type39;
    typedef typename __combined<__type30,__type39>::type __type40;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type37>::__ptype1 __type41;
    typedef indexable_container<__type6, typename std::remove_reference<__type41>::type> __type42;
    typedef typename __combined<__type37,__type42>::type __type43;
    typedef typename cu_basis_funs_1st_der::type<__type34, __type35, __type21, __type43>::__ptype2 __type44;
    typedef indexable_container<__type6, typename std::remove_reference<__type44>::type> __type45;
    typedef typename __combined<__type5,__type42,__type45,__type37>::type __type46;
    typedef typename pythonic::assignable<__type46>::type __type47;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type48;
    typedef __type48 __type49;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type49>::type>::type __type50;
    typedef typename pythonic::assignable<__type50>::type __type51;
    typedef __type51 __type52;
    typedef indexable_container<__type6, typename std::remove_reference<__type50>::type> __type53;
    typedef typename __combined<__type48,__type53>::type __type54;
    typedef __type54 __type55;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type55>::type>::type __type56;
    typedef indexable_container<__type14, typename std::remove_reference<__type56>::type> __type57;
    typedef typename __combined<__type48,__type53,__type57>::type __type58;
    typedef typename pythonic::assignable<__type56>::type __type59;
    typedef __type59 __type60;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type61;
    typedef __type61 __type62;
    typedef decltype(std::declval<__type22>()(std::declval<__type62>())) __type63;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type63>::type::iterator>::value_type>::type __type64;
    typedef __type64 __type65;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type65>::type>::type __type66;
    typedef decltype(std::declval<__type7>()(std::declval<__type52>(), std::declval<__type60>(), std::declval<__type66>())) __type67;
    typedef typename pythonic::assignable<__type67>::type __type68;
    typedef __type68 __type69;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type69>::type>::type __type70;
    typedef typename pythonic::assignable<__type70>::type __type71;
    typedef __type71 __type72;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type69>::type>::type __type73;
    typedef typename cu_basis_funs::type<__type72, __type73, __type37>::__ptype21 __type75;
    typedef indexable_container<__type14, typename std::remove_reference<__type75>::type> __type76;
    typedef typename __combined<__type68,__type76>::type __type77;
    typedef typename cu_basis_funs::type<__type72, __type73, __type37>::__ptype22 __type78;
    typedef indexable_container<__type6, typename std::remove_reference<__type78>::type> __type79;
    typedef typename __combined<__type37,__type79>::type __type80;
    typedef typename cu_basis_funs::type<__type72, __type73, __type80>::__ptype23 __type81;
    typedef indexable_container<__type6, typename std::remove_reference<__type81>::type> __type82;
    typedef typename __combined<__type5,__type79,__type82,__type37>::type __type83;
    typedef typename pythonic::assignable<__type83>::type __type84;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type85;
    typedef decltype(std::declval<__type0>()(std::declval<__type85>(), std::declval<__type3>())) __type86;
    typedef typename pythonic::assignable<__type86>::type __type87;
    typedef long __type88;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type88>(), std::declval<__type88>())) __type89;
    typedef indexable<__type89> __type90;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type91;
    typedef __type91 __type92;
    typedef pythonic::types::contiguous_slice __type93;
    typedef decltype(std::declval<__type92>()(std::declval<__type93>(), std::declval<__type93>())) __type94;
    typedef container<typename std::remove_reference<__type94>::type> __type95;
    typedef typename __combined<__type87,__type95>::type __type96;
    typedef __type96 __type97;
    typedef decltype(std::declval<__type97>()[std::declval<__type89>()]) __type98;
    typedef __type83 __type99;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type99>::type>::type __type100;
    typedef decltype(pythonic::operator_::mul(std::declval<__type98>(), std::declval<__type100>())) __type101;
    typedef container<typename std::remove_reference<__type101>::type> __type102;
    typedef typename __combined<__type87,__type90,__type95,__type102>::type __type103;
    typedef __type103 __type104;
    typedef decltype(std::declval<__type104>()[std::declval<__type89>()]) __type105;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type99>::type>::type __type107;
    typedef decltype(pythonic::operator_::mul(std::declval<__type105>(), std::declval<__type107>())) __type108;
    typedef container<typename std::remove_reference<__type108>::type> __type109;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109>::type __type110;
    typedef __type110 __type111;
    typedef decltype(std::declval<__type111>()[std::declval<__type89>()]) __type112;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type99>::type>::type __type114;
    typedef decltype(pythonic::operator_::mul(std::declval<__type112>(), std::declval<__type114>())) __type115;
    typedef container<typename std::remove_reference<__type115>::type> __type116;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116>::type __type117;
    typedef __type117 __type118;
    typedef decltype(std::declval<__type118>()[std::declval<__type89>()]) __type119;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type99>::type>::type __type121;
    typedef decltype(pythonic::operator_::mul(std::declval<__type119>(), std::declval<__type121>())) __type122;
    typedef container<typename std::remove_reference<__type122>::type> __type123;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123>::type __type124;
    typedef __type124 __type125;
    typedef decltype(std::declval<__type125>()[std::declval<__type89>()]) __type126;
    typedef decltype(pythonic::operator_::mul(std::declval<__type126>(), std::declval<__type100>())) __type129;
    typedef container<typename std::remove_reference<__type129>::type> __type130;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130>::type __type131;
    typedef __type131 __type132;
    typedef decltype(std::declval<__type132>()[std::declval<__type89>()]) __type133;
    typedef decltype(pythonic::operator_::mul(std::declval<__type133>(), std::declval<__type107>())) __type136;
    typedef container<typename std::remove_reference<__type136>::type> __type137;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137>::type __type138;
    typedef __type138 __type139;
    typedef decltype(std::declval<__type139>()[std::declval<__type89>()]) __type140;
    typedef decltype(pythonic::operator_::mul(std::declval<__type140>(), std::declval<__type114>())) __type143;
    typedef container<typename std::remove_reference<__type143>::type> __type144;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144>::type __type145;
    typedef __type145 __type146;
    typedef decltype(std::declval<__type146>()[std::declval<__type89>()]) __type147;
    typedef decltype(pythonic::operator_::mul(std::declval<__type147>(), std::declval<__type121>())) __type150;
    typedef container<typename std::remove_reference<__type150>::type> __type151;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151>::type __type152;
    typedef __type152 __type153;
    typedef decltype(std::declval<__type153>()[std::declval<__type89>()]) __type154;
    typedef decltype(pythonic::operator_::mul(std::declval<__type154>(), std::declval<__type100>())) __type157;
    typedef container<typename std::remove_reference<__type157>::type> __type158;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158>::type __type159;
    typedef __type159 __type160;
    typedef decltype(std::declval<__type160>()[std::declval<__type89>()]) __type161;
    typedef decltype(pythonic::operator_::mul(std::declval<__type161>(), std::declval<__type107>())) __type164;
    typedef container<typename std::remove_reference<__type164>::type> __type165;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165>::type __type166;
    typedef __type166 __type167;
    typedef decltype(std::declval<__type167>()[std::declval<__type89>()]) __type168;
    typedef decltype(pythonic::operator_::mul(std::declval<__type168>(), std::declval<__type114>())) __type171;
    typedef container<typename std::remove_reference<__type171>::type> __type172;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172>::type __type173;
    typedef __type173 __type174;
    typedef decltype(std::declval<__type174>()[std::declval<__type89>()]) __type175;
    typedef decltype(pythonic::operator_::mul(std::declval<__type175>(), std::declval<__type121>())) __type178;
    typedef container<typename std::remove_reference<__type178>::type> __type179;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179>::type __type180;
    typedef __type180 __type181;
    typedef decltype(std::declval<__type181>()[std::declval<__type89>()]) __type182;
    typedef decltype(pythonic::operator_::mul(std::declval<__type182>(), std::declval<__type100>())) __type185;
    typedef container<typename std::remove_reference<__type185>::type> __type186;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186>::type __type187;
    typedef __type187 __type188;
    typedef decltype(std::declval<__type188>()[std::declval<__type89>()]) __type189;
    typedef decltype(pythonic::operator_::mul(std::declval<__type189>(), std::declval<__type107>())) __type192;
    typedef container<typename std::remove_reference<__type192>::type> __type193;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193>::type __type194;
    typedef __type194 __type195;
    typedef decltype(std::declval<__type195>()[std::declval<__type89>()]) __type196;
    typedef decltype(pythonic::operator_::mul(std::declval<__type196>(), std::declval<__type114>())) __type199;
    typedef container<typename std::remove_reference<__type199>::type> __type200;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200>::type __type201;
    typedef __type201 __type202;
    typedef decltype(std::declval<__type202>()[std::declval<__type89>()]) __type203;
    typedef decltype(pythonic::operator_::mul(std::declval<__type203>(), std::declval<__type121>())) __type206;
    typedef container<typename std::remove_reference<__type206>::type> __type207;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200,__type207>::type __type208;
    typedef typename pythonic::assignable<__type208>::type __type209;
    typedef typename pythonic::assignable<__type40>::type __type210;
    typedef typename pythonic::assignable<__type77>::type __type211;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type47 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type84 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type209 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      for (auto&& __tuple0: pythonic::builtins::functor::enumerate{}(X))
      {
        typename pythonic::lazy<decltype(std::get<0>(__tuple0))>::type i = std::get<0>(__tuple0);
        __type210 __tuple1 = cu_find_span()(xmin, dx, std::get<1>(__tuple0));
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span1 = std::get<0>(__tuple1);
        cu_basis_funs_1st_der()(span1, std::get<1>(__tuple1), dx, basis1);
        {
          for (auto&& __tuple2: pythonic::builtins::functor::enumerate{}(Y))
          {
            typename pythonic::lazy<decltype(std::get<0>(__tuple2))>::type j = std::get<0>(__tuple2);
            __type211 __tuple3 = cu_find_span()(ymin, dy, std::get<1>(__tuple2));
            typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple3))>::type span2 = std::get<0>(__tuple3);
            cu_basis_funs()(span2, std::get<1>(__tuple3), basis2);
            theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
            z[pythonic::types::make_tuple(i, j)] = 0.0;
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 >
  inline
  typename cu_eval_spline_2d_cross_01::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7>::result_type cu_eval_spline_2d_cross_01::operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::builtins::functor::enumerate{})>::type>::type __type22;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type23;
    typedef __type23 __type24;
    typedef decltype(std::declval<__type22>()(std::declval<__type24>())) __type25;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type25>::type::iterator>::value_type>::type __type26;
    typedef __type26 __type27;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type27>::type>::type __type28;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type28>())) __type29;
    typedef typename pythonic::assignable<__type29>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename pythonic::assignable<__type32>::type __type33;
    typedef __type33 __type34;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type35;
    typedef __type5 __type36;
    typedef typename cu_basis_funs::type<__type34, __type35, __type36>::__ptype21 __type37;
    typedef indexable_container<__type14, typename std::remove_reference<__type37>::type> __type38;
    typedef typename __combined<__type30,__type38>::type __type39;
    typedef typename cu_basis_funs::type<__type34, __type35, __type36>::__ptype22 __type40;
    typedef indexable_container<__type6, typename std::remove_reference<__type40>::type> __type41;
    typedef typename __combined<__type36,__type41>::type __type42;
    typedef typename cu_basis_funs::type<__type34, __type35, __type42>::__ptype23 __type43;
    typedef indexable_container<__type6, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type5,__type41,__type44,__type36>::type __type45;
    typedef typename pythonic::assignable<__type45>::type __type46;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type47;
    typedef __type47 __type48;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type48>::type>::type __type49;
    typedef typename pythonic::assignable<__type49>::type __type50;
    typedef __type50 __type51;
    typedef indexable_container<__type6, typename std::remove_reference<__type49>::type> __type52;
    typedef typename __combined<__type47,__type52>::type __type53;
    typedef __type53 __type54;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type54>::type>::type __type55;
    typedef indexable_container<__type14, typename std::remove_reference<__type55>::type> __type56;
    typedef typename __combined<__type47,__type52,__type56>::type __type57;
    typedef typename pythonic::assignable<__type55>::type __type58;
    typedef __type58 __type59;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type60;
    typedef __type60 __type61;
    typedef decltype(std::declval<__type22>()(std::declval<__type61>())) __type62;
    typedef typename std::remove_cv<typename std::iterator_traits<typename std::remove_reference<__type62>::type::iterator>::value_type>::type __type63;
    typedef __type63 __type64;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type64>::type>::type __type65;
    typedef decltype(std::declval<__type7>()(std::declval<__type51>(), std::declval<__type59>(), std::declval<__type65>())) __type66;
    typedef typename pythonic::assignable<__type66>::type __type67;
    typedef __type67 __type68;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type68>::type>::type __type69;
    typedef typename pythonic::assignable<__type69>::type __type70;
    typedef __type70 __type71;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type68>::type>::type __type72;
    typedef typename cu_basis_funs_1st_der::type<__type71, __type72, __type21, __type36>::__ptype0 __type75;
    typedef indexable_container<__type14, typename std::remove_reference<__type75>::type> __type76;
    typedef typename __combined<__type67,__type76>::type __type77;
    typedef typename cu_basis_funs_1st_der::type<__type71, __type72, __type21, __type36>::__ptype1 __type78;
    typedef indexable_container<__type6, typename std::remove_reference<__type78>::type> __type79;
    typedef typename __combined<__type36,__type79>::type __type80;
    typedef typename cu_basis_funs_1st_der::type<__type71, __type72, __type21, __type80>::__ptype2 __type81;
    typedef indexable_container<__type6, typename std::remove_reference<__type81>::type> __type82;
    typedef typename __combined<__type5,__type79,__type82,__type36>::type __type83;
    typedef typename pythonic::assignable<__type83>::type __type84;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>(), std::declval<__type2>())) __type85;
    typedef decltype(std::declval<__type0>()(std::declval<__type85>(), std::declval<__type3>())) __type86;
    typedef typename pythonic::assignable<__type86>::type __type87;
    typedef long __type88;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type88>(), std::declval<__type88>())) __type89;
    typedef indexable<__type89> __type90;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type91;
    typedef __type91 __type92;
    typedef pythonic::types::contiguous_slice __type93;
    typedef decltype(std::declval<__type92>()(std::declval<__type93>(), std::declval<__type93>())) __type94;
    typedef container<typename std::remove_reference<__type94>::type> __type95;
    typedef typename __combined<__type87,__type95>::type __type96;
    typedef __type96 __type97;
    typedef decltype(std::declval<__type97>()[std::declval<__type89>()]) __type98;
    typedef __type83 __type99;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type99>::type>::type __type100;
    typedef decltype(pythonic::operator_::mul(std::declval<__type98>(), std::declval<__type100>())) __type101;
    typedef container<typename std::remove_reference<__type101>::type> __type102;
    typedef typename __combined<__type87,__type90,__type95,__type102>::type __type103;
    typedef __type103 __type104;
    typedef decltype(std::declval<__type104>()[std::declval<__type89>()]) __type105;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type99>::type>::type __type107;
    typedef decltype(pythonic::operator_::mul(std::declval<__type105>(), std::declval<__type107>())) __type108;
    typedef container<typename std::remove_reference<__type108>::type> __type109;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109>::type __type110;
    typedef __type110 __type111;
    typedef decltype(std::declval<__type111>()[std::declval<__type89>()]) __type112;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type99>::type>::type __type114;
    typedef decltype(pythonic::operator_::mul(std::declval<__type112>(), std::declval<__type114>())) __type115;
    typedef container<typename std::remove_reference<__type115>::type> __type116;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116>::type __type117;
    typedef __type117 __type118;
    typedef decltype(std::declval<__type118>()[std::declval<__type89>()]) __type119;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type99>::type>::type __type121;
    typedef decltype(pythonic::operator_::mul(std::declval<__type119>(), std::declval<__type121>())) __type122;
    typedef container<typename std::remove_reference<__type122>::type> __type123;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123>::type __type124;
    typedef __type124 __type125;
    typedef decltype(std::declval<__type125>()[std::declval<__type89>()]) __type126;
    typedef decltype(pythonic::operator_::mul(std::declval<__type126>(), std::declval<__type100>())) __type129;
    typedef container<typename std::remove_reference<__type129>::type> __type130;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130>::type __type131;
    typedef __type131 __type132;
    typedef decltype(std::declval<__type132>()[std::declval<__type89>()]) __type133;
    typedef decltype(pythonic::operator_::mul(std::declval<__type133>(), std::declval<__type107>())) __type136;
    typedef container<typename std::remove_reference<__type136>::type> __type137;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137>::type __type138;
    typedef __type138 __type139;
    typedef decltype(std::declval<__type139>()[std::declval<__type89>()]) __type140;
    typedef decltype(pythonic::operator_::mul(std::declval<__type140>(), std::declval<__type114>())) __type143;
    typedef container<typename std::remove_reference<__type143>::type> __type144;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144>::type __type145;
    typedef __type145 __type146;
    typedef decltype(std::declval<__type146>()[std::declval<__type89>()]) __type147;
    typedef decltype(pythonic::operator_::mul(std::declval<__type147>(), std::declval<__type121>())) __type150;
    typedef container<typename std::remove_reference<__type150>::type> __type151;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151>::type __type152;
    typedef __type152 __type153;
    typedef decltype(std::declval<__type153>()[std::declval<__type89>()]) __type154;
    typedef decltype(pythonic::operator_::mul(std::declval<__type154>(), std::declval<__type100>())) __type157;
    typedef container<typename std::remove_reference<__type157>::type> __type158;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158>::type __type159;
    typedef __type159 __type160;
    typedef decltype(std::declval<__type160>()[std::declval<__type89>()]) __type161;
    typedef decltype(pythonic::operator_::mul(std::declval<__type161>(), std::declval<__type107>())) __type164;
    typedef container<typename std::remove_reference<__type164>::type> __type165;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165>::type __type166;
    typedef __type166 __type167;
    typedef decltype(std::declval<__type167>()[std::declval<__type89>()]) __type168;
    typedef decltype(pythonic::operator_::mul(std::declval<__type168>(), std::declval<__type114>())) __type171;
    typedef container<typename std::remove_reference<__type171>::type> __type172;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172>::type __type173;
    typedef __type173 __type174;
    typedef decltype(std::declval<__type174>()[std::declval<__type89>()]) __type175;
    typedef decltype(pythonic::operator_::mul(std::declval<__type175>(), std::declval<__type121>())) __type178;
    typedef container<typename std::remove_reference<__type178>::type> __type179;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179>::type __type180;
    typedef __type180 __type181;
    typedef decltype(std::declval<__type181>()[std::declval<__type89>()]) __type182;
    typedef decltype(pythonic::operator_::mul(std::declval<__type182>(), std::declval<__type100>())) __type185;
    typedef container<typename std::remove_reference<__type185>::type> __type186;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186>::type __type187;
    typedef __type187 __type188;
    typedef decltype(std::declval<__type188>()[std::declval<__type89>()]) __type189;
    typedef decltype(pythonic::operator_::mul(std::declval<__type189>(), std::declval<__type107>())) __type192;
    typedef container<typename std::remove_reference<__type192>::type> __type193;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193>::type __type194;
    typedef __type194 __type195;
    typedef decltype(std::declval<__type195>()[std::declval<__type89>()]) __type196;
    typedef decltype(pythonic::operator_::mul(std::declval<__type196>(), std::declval<__type114>())) __type199;
    typedef container<typename std::remove_reference<__type199>::type> __type200;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200>::type __type201;
    typedef __type201 __type202;
    typedef decltype(std::declval<__type202>()[std::declval<__type89>()]) __type203;
    typedef decltype(pythonic::operator_::mul(std::declval<__type203>(), std::declval<__type121>())) __type206;
    typedef container<typename std::remove_reference<__type206>::type> __type207;
    typedef typename __combined<__type87,__type90,__type95,__type102,__type109,__type116,__type123,__type130,__type137,__type144,__type151,__type158,__type165,__type172,__type179,__type186,__type193,__type200,__type207>::type __type208;
    typedef typename pythonic::assignable<__type208>::type __type209;
    typedef typename pythonic::assignable<__type39>::type __type210;
    typedef typename pythonic::assignable<__type77>::type __type211;
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts1))>::type xmin = std::get<0>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    typename pythonic::assignable_noescape<decltype(std::get<0>(kts2))>::type ymin = std::get<0>(kts2);
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts2))>::type dy = std::get<1>(kts2);
    __type46 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type84 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type209 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    {
      for (auto&& __tuple0: pythonic::builtins::functor::enumerate{}(X))
      {
        typename pythonic::lazy<decltype(std::get<0>(__tuple0))>::type i = std::get<0>(__tuple0);
        __type210 __tuple1 = cu_find_span()(xmin, dx, std::get<1>(__tuple0));
        typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span1 = std::get<0>(__tuple1);
        cu_basis_funs()(span1, std::get<1>(__tuple1), basis1);
        {
          for (auto&& __tuple2: pythonic::builtins::functor::enumerate{}(Y))
          {
            typename pythonic::lazy<decltype(std::get<0>(__tuple2))>::type j = std::get<0>(__tuple2);
            __type211 __tuple3 = cu_find_span()(ymin, dy, std::get<1>(__tuple2));
            typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple3))>::type span2 = std::get<0>(__tuple3);
            cu_basis_funs_1st_der()(span2, std::get<1>(__tuple3), dx, basis2);
            theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
            z[pythonic::types::make_tuple(i, j)] = 0.0;
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
            theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
            z[pythonic::types::make_tuple(i, j)] += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 >
  inline
  typename cu_eval_spline_2d_scalar::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8>::result_type cu_eval_spline_2d_scalar::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7 der1, argument_type8 der2) const
  {
    typedef cu_find_span __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type1;
    typedef std::integral_constant<long,1> __type2;
    typedef __type1 __type3;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type3>::type>::type __type4;
    typedef indexable_container<__type2, typename std::remove_reference<__type4>::type> __type5;
    typedef typename __combined<__type1,__type5>::type __type6;
    typedef __type6 __type7;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type7>::type>::type __type8;
    typedef typename pythonic::assignable<__type4>::type __type9;
    typedef __type9 __type10;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type11;
    typedef __type11 __type12;
    typedef decltype(std::declval<__type0>()(std::declval<__type8>(), std::declval<__type10>(), std::declval<__type12>())) __type13;
    typedef typename pythonic::assignable<__type13>::type __type14;
    typedef __type14 __type15;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type15>::type>::type __type16;
    typedef typename pythonic::assignable<__type16>::type __type17;
    typedef __type17 __type18;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type15>::type>::type __type19;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type20;
    typedef double __type21;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type21>(), std::declval<__type21>(), std::declval<__type21>(), std::declval<__type21>())) __type22;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type23;
    typedef decltype(std::declval<__type20>()(std::declval<__type22>(), std::declval<__type23>())) __type24;
    typedef typename pythonic::assignable<__type24>::type __type25;
    typedef __type25 __type26;
    typedef typename cu_basis_funs::type<__type18, __type19, __type26>::__ptype21 __type27;
    typedef indexable_container<__type2, typename std::remove_reference<__type27>::type> __type28;
    typedef typename __combined<__type14,__type28>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename __combined<__type19,__type32>::type __type33;
    typedef std::integral_constant<long,0> __type35;
    typedef typename cu_basis_funs::type<__type18, __type19, __type26>::__ptype22 __type36;
    typedef indexable_container<__type35, typename std::remove_reference<__type36>::type> __type37;
    typedef typename __combined<__type26,__type37>::type __type38;
    typedef typename cu_basis_funs::type<__type18, __type19, __type38>::__ptype23 __type39;
    typedef indexable_container<__type35, typename std::remove_reference<__type39>::type> __type40;
    typedef typename __combined<__type25,__type37,__type40,__type26>::type __type41;
    typedef __type41 __type42;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type42>::__ptype0 __type43;
    typedef indexable_container<__type2, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type14,__type28,__type44>::type __type45;
    typedef typename pythonic::assignable<__type45>::type __type46;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type47;
    typedef __type47 __type48;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type48>::type>::type __type49;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type48>::type>::type __type50;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type51;
    typedef __type51 __type52;
    typedef decltype(std::declval<__type0>()(std::declval<__type49>(), std::declval<__type50>(), std::declval<__type52>())) __type53;
    typedef typename pythonic::assignable<__type53>::type __type54;
    typedef __type54 __type55;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type55>::type>::type __type56;
    typedef typename pythonic::assignable<__type56>::type __type57;
    typedef __type57 __type58;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type55>::type>::type __type59;
    typedef typename cu_basis_funs::type<__type58, __type59, __type26>::__ptype21 __type61;
    typedef indexable_container<__type2, typename std::remove_reference<__type61>::type> __type62;
    typedef typename __combined<__type54,__type62>::type __type64;
    typedef __type64 __type65;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type65>::type>::type __type66;
    typedef typename __combined<__type59,__type66>::type __type67;
    typedef typename cu_basis_funs::type<__type58, __type59, __type26>::__ptype22 __type69;
    typedef indexable_container<__type35, typename std::remove_reference<__type69>::type> __type70;
    typedef typename __combined<__type26,__type70>::type __type71;
    typedef typename cu_basis_funs::type<__type58, __type59, __type71>::__ptype23 __type72;
    typedef indexable_container<__type35, typename std::remove_reference<__type72>::type> __type73;
    typedef typename __combined<__type25,__type70,__type73,__type26>::type __type74;
    typedef __type74 __type75;
    typedef typename cu_basis_funs_1st_der::type<__type58, __type67, __type10, __type75>::__ptype0 __type76;
    typedef indexable_container<__type2, typename std::remove_reference<__type76>::type> __type77;
    typedef typename __combined<__type54,__type62,__type77>::type __type78;
    typedef typename pythonic::assignable<__type78>::type __type79;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type42>::__ptype1 __type80;
    typedef indexable_container<__type35, typename std::remove_reference<__type80>::type> __type81;
    typedef typename __combined<__type42,__type81>::type __type82;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type82>::__ptype2 __type83;
    typedef indexable_container<__type35, typename std::remove_reference<__type83>::type> __type84;
    typedef typename __combined<__type25,__type37,__type40,__type81,__type84,__type26,__type42>::type __type85;
    typedef typename pythonic::assignable<__type85>::type __type86;
    typedef typename cu_basis_funs_1st_der::type<__type58, __type67, __type10, __type75>::__ptype1 __type87;
    typedef indexable_container<__type35, typename std::remove_reference<__type87>::type> __type88;
    typedef typename __combined<__type75,__type88>::type __type89;
    typedef typename cu_basis_funs_1st_der::type<__type58, __type67, __type10, __type89>::__ptype2 __type90;
    typedef indexable_container<__type35, typename std::remove_reference<__type90>::type> __type91;
    typedef typename __combined<__type25,__type70,__type73,__type88,__type91,__type26,__type75>::type __type92;
    typedef typename pythonic::assignable<__type92>::type __type93;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type22>(), std::declval<__type22>(), std::declval<__type22>(), std::declval<__type22>())) __type94;
    typedef decltype(std::declval<__type20>()(std::declval<__type94>(), std::declval<__type23>())) __type95;
    typedef typename pythonic::assignable<__type95>::type __type96;
    typedef long __type97;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type97>(), std::declval<__type97>())) __type98;
    typedef indexable<__type98> __type99;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type100;
    typedef __type100 __type101;
    typedef pythonic::types::contiguous_slice __type102;
    typedef decltype(std::declval<__type101>()(std::declval<__type102>(), std::declval<__type102>())) __type103;
    typedef container<typename std::remove_reference<__type103>::type> __type104;
    typedef typename __combined<__type96,__type104>::type __type105;
    typedef __type105 __type106;
    typedef decltype(std::declval<__type106>()[std::declval<__type98>()]) __type107;
    typedef __type92 __type108;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type108>::type>::type __type109;
    typedef decltype(pythonic::operator_::mul(std::declval<__type107>(), std::declval<__type109>())) __type110;
    typedef container<typename std::remove_reference<__type110>::type> __type111;
    typedef typename __combined<__type96,__type99,__type104,__type111>::type __type112;
    typedef __type112 __type113;
    typedef decltype(std::declval<__type113>()[std::declval<__type98>()]) __type114;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type108>::type>::type __type116;
    typedef decltype(pythonic::operator_::mul(std::declval<__type114>(), std::declval<__type116>())) __type117;
    typedef container<typename std::remove_reference<__type117>::type> __type118;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118>::type __type119;
    typedef __type119 __type120;
    typedef decltype(std::declval<__type120>()[std::declval<__type98>()]) __type121;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type108>::type>::type __type123;
    typedef decltype(pythonic::operator_::mul(std::declval<__type121>(), std::declval<__type123>())) __type124;
    typedef container<typename std::remove_reference<__type124>::type> __type125;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125>::type __type126;
    typedef __type126 __type127;
    typedef decltype(std::declval<__type127>()[std::declval<__type98>()]) __type128;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type108>::type>::type __type130;
    typedef decltype(pythonic::operator_::mul(std::declval<__type128>(), std::declval<__type130>())) __type131;
    typedef container<typename std::remove_reference<__type131>::type> __type132;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132>::type __type133;
    typedef __type133 __type134;
    typedef decltype(std::declval<__type134>()[std::declval<__type98>()]) __type135;
    typedef decltype(pythonic::operator_::mul(std::declval<__type135>(), std::declval<__type109>())) __type138;
    typedef container<typename std::remove_reference<__type138>::type> __type139;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139>::type __type140;
    typedef __type140 __type141;
    typedef decltype(std::declval<__type141>()[std::declval<__type98>()]) __type142;
    typedef decltype(pythonic::operator_::mul(std::declval<__type142>(), std::declval<__type116>())) __type145;
    typedef container<typename std::remove_reference<__type145>::type> __type146;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146>::type __type147;
    typedef __type147 __type148;
    typedef decltype(std::declval<__type148>()[std::declval<__type98>()]) __type149;
    typedef decltype(pythonic::operator_::mul(std::declval<__type149>(), std::declval<__type123>())) __type152;
    typedef container<typename std::remove_reference<__type152>::type> __type153;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153>::type __type154;
    typedef __type154 __type155;
    typedef decltype(std::declval<__type155>()[std::declval<__type98>()]) __type156;
    typedef decltype(pythonic::operator_::mul(std::declval<__type156>(), std::declval<__type130>())) __type159;
    typedef container<typename std::remove_reference<__type159>::type> __type160;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160>::type __type161;
    typedef __type161 __type162;
    typedef decltype(std::declval<__type162>()[std::declval<__type98>()]) __type163;
    typedef decltype(pythonic::operator_::mul(std::declval<__type163>(), std::declval<__type109>())) __type166;
    typedef container<typename std::remove_reference<__type166>::type> __type167;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167>::type __type168;
    typedef __type168 __type169;
    typedef decltype(std::declval<__type169>()[std::declval<__type98>()]) __type170;
    typedef decltype(pythonic::operator_::mul(std::declval<__type170>(), std::declval<__type116>())) __type173;
    typedef container<typename std::remove_reference<__type173>::type> __type174;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174>::type __type175;
    typedef __type175 __type176;
    typedef decltype(std::declval<__type176>()[std::declval<__type98>()]) __type177;
    typedef decltype(pythonic::operator_::mul(std::declval<__type177>(), std::declval<__type123>())) __type180;
    typedef container<typename std::remove_reference<__type180>::type> __type181;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181>::type __type182;
    typedef __type182 __type183;
    typedef decltype(std::declval<__type183>()[std::declval<__type98>()]) __type184;
    typedef decltype(pythonic::operator_::mul(std::declval<__type184>(), std::declval<__type130>())) __type187;
    typedef container<typename std::remove_reference<__type187>::type> __type188;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188>::type __type189;
    typedef __type189 __type190;
    typedef decltype(std::declval<__type190>()[std::declval<__type98>()]) __type191;
    typedef decltype(pythonic::operator_::mul(std::declval<__type191>(), std::declval<__type109>())) __type194;
    typedef container<typename std::remove_reference<__type194>::type> __type195;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195>::type __type196;
    typedef __type196 __type197;
    typedef decltype(std::declval<__type197>()[std::declval<__type98>()]) __type198;
    typedef decltype(pythonic::operator_::mul(std::declval<__type198>(), std::declval<__type116>())) __type201;
    typedef container<typename std::remove_reference<__type201>::type> __type202;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195,__type202>::type __type203;
    typedef __type203 __type204;
    typedef decltype(std::declval<__type204>()[std::declval<__type98>()]) __type205;
    typedef decltype(pythonic::operator_::mul(std::declval<__type205>(), std::declval<__type123>())) __type208;
    typedef container<typename std::remove_reference<__type208>::type> __type209;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195,__type202,__type209>::type __type210;
    typedef __type210 __type211;
    typedef decltype(std::declval<__type211>()[std::declval<__type98>()]) __type212;
    typedef decltype(pythonic::operator_::mul(std::declval<__type212>(), std::declval<__type130>())) __type215;
    typedef container<typename std::remove_reference<__type215>::type> __type216;
    typedef typename __combined<__type96,__type99,__type104,__type111,__type118,__type125,__type132,__type139,__type146,__type153,__type160,__type167,__type174,__type181,__type188,__type195,__type202,__type209,__type216>::type __type217;
    typedef typename pythonic::assignable<__type217>::type __type218;
    typedef typename pythonic::assignable<__type21>::type __type219;
    typedef __type217 __type220;
    typedef decltype(std::declval<__type220>()[std::declval<__type98>()]) __type221;
    typedef __type85 __type222;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type222>::type>::type __type223;
    typedef decltype(pythonic::operator_::mul(std::declval<__type221>(), std::declval<__type223>())) __type224;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type222>::type>::type __type228;
    typedef decltype(pythonic::operator_::mul(std::declval<__type221>(), std::declval<__type228>())) __type229;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type222>::type>::type __type233;
    typedef decltype(pythonic::operator_::mul(std::declval<__type221>(), std::declval<__type233>())) __type234;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type222>::type>::type __type238;
    typedef decltype(pythonic::operator_::mul(std::declval<__type221>(), std::declval<__type238>())) __type239;
    typedef typename __combined<__type219,__type224,__type229,__type234,__type239>::type __type240;
    typedef typename pythonic::assignable<__type240>::type __type241;
    typename pythonic::assignable_noescape<decltype(std::get<1>(kts1))>::type dx = std::get<1>(kts1);
    __type46 __tuple0 = cu_find_span()(std::get<0>(kts1), dx, x);
    typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span1 = std::get<0>(__tuple0);
    __type79 __tuple1 = cu_find_span()(std::get<0>(kts2), std::get<1>(kts2), y);
    typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span2 = std::get<0>(__tuple1);
    __type86 basis1 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    __type93 basis2 = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    if (pythonic::operator_::eq(der1, 0L))
    {
      cu_basis_funs()(span1, std::get<1>(__tuple0), basis1);
    }
    else
    {
      if (pythonic::operator_::eq(der1, 1L))
      {
        cu_basis_funs_1st_der()(span1, std::get<1>(__tuple0), dx, basis1);
      }
    }
    if (pythonic::operator_::eq(der2, 0L))
    {
      cu_basis_funs()(span2, std::get<1>(__tuple1), basis2);
    }
    else
    {
      if (pythonic::operator_::eq(der2, 1L))
      {
        cu_basis_funs_1st_der()(span2, std::get<1>(__tuple1), dx, basis2);
      }
    }
    __type218 theCoeffs = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(pythonic::types::make_tuple(1.92827656e-316, 0.0, 1.95663513e-316, 6.93169476233714e-310), pythonic::types::make_tuple(4.976231423109604e-270, 4.739620985876225e-270, 1.731569228383845e-260, 1.731569228336522e-260), pythonic::types::make_tuple(4.976224230318305e-270, 6.159337765603442e-270, 9.348079238081933e-222, 2.759264062999405e-306), pythonic::types::make_tuple(9.952498737896141e-270, 1.1372753677482506e-269, 1.8122608642614116e-308, -3.184562956816477e+136)), pythonic::numpy::functor::float64{});
    theCoeffs(pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None),pythonic::types::contiguous_slice(pythonic::builtins::None,pythonic::builtins::None)) = coeffs(pythonic::types::contiguous_slice(pythonic::operator_::sub(span1, deg1),pythonic::operator_::add(span1, 1L)),pythonic::types::contiguous_slice(pythonic::operator_::sub(span2, deg2),pythonic::operator_::add(span2, 1L)));
    __type241 z = 0.0;
    theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 1L)), std::get<1>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 2L)), std::get<2>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 3L)), std::get<3>(basis2));
    z += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(0L, 0L)), std::get<0>(basis1));
    theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<0>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 1L)), std::get<1>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 2L)), std::get<2>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 3L)), std::get<3>(basis2));
    z += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(1L, 0L)), std::get<1>(basis1));
    theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<0>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 1L)), std::get<1>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 2L)), std::get<2>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 3L)), std::get<3>(basis2));
    z += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(2L, 0L)), std::get<2>(basis1));
    theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) = pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<0>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 1L)), std::get<1>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 2L)), std::get<2>(basis2));
    theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)) += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 3L)), std::get<3>(basis2));
    z += pythonic::operator_::mul(theCoeffs.fast(pythonic::types::make_tuple(3L, 0L)), std::get<3>(basis1));
    return z;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 >
  inline
  typename cu_eval_spline_1d_vector::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5>::result_type cu_eval_spline_1d_vector::operator()(argument_type0&& x, argument_type1&& knots, argument_type2&& degree, argument_type3&& coeffs, argument_type4&& y, argument_type5 der) const
  {
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type0;
    typedef double __type1;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>(), std::declval<__type1>())) __type2;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type3;
    typedef decltype(std::declval<__type0>()(std::declval<__type2>(), std::declval<__type3>())) __type4;
    typedef typename pythonic::assignable<__type4>::type __type5;
    typedef std::integral_constant<long,0> __type6;
    typedef cu_find_span __type7;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type8;
    typedef __type8 __type9;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type9>::type>::type __type10;
    typedef typename pythonic::assignable<__type10>::type __type11;
    typedef __type11 __type12;
    typedef indexable_container<__type6, typename std::remove_reference<__type10>::type> __type13;
    typedef std::integral_constant<long,1> __type14;
    typedef typename __combined<__type8,__type13>::type __type15;
    typedef __type15 __type16;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type16>::type>::type __type17;
    typedef indexable_container<__type14, typename std::remove_reference<__type17>::type> __type18;
    typedef typename __combined<__type8,__type13,__type18>::type __type19;
    typedef typename pythonic::assignable<__type17>::type __type20;
    typedef __type20 __type21;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type22;
    typedef __type22 __type23;
    typedef decltype(std::declval<__type7>()(std::declval<__type12>(), std::declval<__type21>(), std::declval<__type23>())) __type24;
    typedef typename pythonic::assignable<__type24>::type __type25;
    typedef __type25 __type26;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type26>::type>::type __type27;
    typedef typename pythonic::assignable<__type27>::type __type28;
    typedef __type28 __type29;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type26>::type>::type __type30;
    typedef __type5 __type31;
    typedef typename cu_basis_funs::type<__type29, __type30, __type31>::__ptype21 __type32;
    typedef indexable_container<__type14, typename std::remove_reference<__type32>::type> __type33;
    typedef typename __combined<__type25,__type33>::type __type34;
    typedef typename cu_basis_funs::type<__type29, __type30, __type31>::__ptype22 __type35;
    typedef indexable_container<__type6, typename std::remove_reference<__type35>::type> __type36;
    typedef typename __combined<__type31,__type36>::type __type37;
    typedef typename cu_basis_funs::type<__type29, __type30, __type37>::__ptype23 __type38;
    typedef indexable_container<__type6, typename std::remove_reference<__type38>::type> __type39;
    typedef typename pythonic::assignable<__type30>::type __type50;
    typedef __type50 __type51;
    typedef typename __combined<__type5,__type36,__type39,__type31>::type __type52;
    typedef __type52 __type53;
    typedef typename cu_basis_funs::type<__type29, __type51, __type53>::__ptype21 __type54;
    typedef typename __combined<__type51,__type54>::type __type55;
    typedef typename cu_basis_funs::type<__type29, __type55, __type53>::__ptype22 __type56;
    typedef indexable_container<__type6, typename std::remove_reference<__type56>::type> __type57;
    typedef typename __combined<__type53,__type57>::type __type58;
    typedef typename cu_basis_funs::type<__type29, __type55, __type58>::__ptype23 __type59;
    typedef indexable_container<__type6, typename std::remove_reference<__type59>::type> __type60;
    typedef typename __combined<__type50,__type54,__type51>::type __type62;
    typedef __type62 __type63;
    typedef typename __combined<__type5,__type36,__type39,__type57,__type60,__type31,__type53>::type __type65;
    typedef __type65 __type66;
    typedef typename cu_basis_funs_1st_der::type<__type29, __type63, __type21, __type66>::__ptype0 __type67;
    typedef typename __combined<__type63,__type67>::type __type68;
    typedef typename cu_basis_funs_1st_der::type<__type29, __type68, __type21, __type66>::__ptype1 __type69;
    typedef indexable_container<__type6, typename std::remove_reference<__type69>::type> __type70;
    typedef typename __combined<__type66,__type70>::type __type71;
    typedef typename cu_basis_funs_1st_der::type<__type29, __type68, __type21, __type71>::__ptype2 __type72;
    typedef indexable_container<__type6, typename std::remove_reference<__type72>::type> __type73;
    typedef typename __combined<__type5,__type36,__type39,__type57,__type60,__type70,__type73,__type31,__type53,__type66>::type __type74;
    typedef typename pythonic::assignable<__type74>::type __type75;
    typedef typename pythonic::assignable<__type34>::type __type76;
    typedef typename __combined<__type50,__type54,__type67,__type51,__type63>::type __type77;
    typedef typename pythonic::assignable<__type77>::type __type78;
    typename pythonic::assignable_noescape<decltype(std::get<0>(knots))>::type xmin = std::get<0>(knots);
    typename pythonic::assignable_noescape<decltype(std::get<1>(knots))>::type dx = std::get<1>(knots);
    __type75 basis = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    if (pythonic::operator_::eq(der, 0L))
    {
      {
        for (auto&& __tuple0: pythonic::builtins::functor::enumerate{}(x))
        {
          typename pythonic::lazy<decltype(std::get<0>(__tuple0))>::type i = std::get<0>(__tuple0);
          __type76 __tuple1 = cu_find_span()(xmin, dx, x);
          typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple1))>::type span = std::get<0>(__tuple1);
          cu_basis_funs()(span, std::get<1>(__tuple1), basis);
          y[i] = 0.0;
          y[i] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 0L)], std::get<0>(basis));
          y[i] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 1L)], std::get<1>(basis));
          y[i] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 2L)], std::get<2>(basis));
          y[i] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 3L)], std::get<3>(basis));
        }
      }
    }
    else
    {
      if (pythonic::operator_::eq(der, 1L))
      {
        {
          for (auto&& __tuple2: pythonic::builtins::functor::enumerate{}(x))
          {
            typename pythonic::lazy<decltype(std::get<0>(__tuple2))>::type i_ = std::get<0>(__tuple2);
            typename pythonic::assignable_noescape<decltype(cu_find_span()(xmin, dx, x))>::type __tuple3 = cu_find_span()(xmin, dx, x);
            typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple3))>::type span_ = std::get<0>(__tuple3);
            __type78 offset_ = std::get<1>(__tuple3);
            cu_basis_funs()(span_, offset_, basis);
            cu_basis_funs_1st_der()(span_, offset_, dx, basis);
            y[i_] = 0.0;
            y[i_] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span_, 3L), 0L)], std::get<0>(basis));
            y[i_] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span_, 3L), 1L)], std::get<1>(basis));
            y[i_] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span_, 3L), 2L)], std::get<2>(basis));
            y[i_] += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span_, 3L), 3L)], std::get<3>(basis));
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
  inline
  typename cu_eval_spline_1d_scalar::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type cu_eval_spline_1d_scalar::operator()(argument_type0&& x, argument_type1&& knots, argument_type2&& degree, argument_type3&& coeffs, argument_type4 der) const
  {
    typedef cu_find_span __type0;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type1;
    typedef std::integral_constant<long,1> __type2;
    typedef __type1 __type3;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type3>::type>::type __type4;
    typedef indexable_container<__type2, typename std::remove_reference<__type4>::type> __type5;
    typedef typename __combined<__type1,__type5>::type __type6;
    typedef __type6 __type7;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type7>::type>::type __type8;
    typedef typename pythonic::assignable<__type4>::type __type9;
    typedef __type9 __type10;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type11;
    typedef __type11 __type12;
    typedef decltype(std::declval<__type0>()(std::declval<__type8>(), std::declval<__type10>(), std::declval<__type12>())) __type13;
    typedef typename pythonic::assignable<__type13>::type __type14;
    typedef __type14 __type15;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type15>::type>::type __type16;
    typedef typename pythonic::assignable<__type16>::type __type17;
    typedef __type17 __type18;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type15>::type>::type __type19;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::array{})>::type>::type __type20;
    typedef double __type21;
    typedef decltype(pythonic::types::make_tuple(std::declval<__type21>(), std::declval<__type21>(), std::declval<__type21>(), std::declval<__type21>())) __type22;
    typedef typename std::remove_cv<typename std::remove_reference<decltype(pythonic::numpy::functor::float64{})>::type>::type __type23;
    typedef decltype(std::declval<__type20>()(std::declval<__type22>(), std::declval<__type23>())) __type24;
    typedef typename pythonic::assignable<__type24>::type __type25;
    typedef __type25 __type26;
    typedef typename cu_basis_funs::type<__type18, __type19, __type26>::__ptype21 __type27;
    typedef indexable_container<__type2, typename std::remove_reference<__type27>::type> __type28;
    typedef typename __combined<__type14,__type28>::type __type30;
    typedef __type30 __type31;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type31>::type>::type __type32;
    typedef typename __combined<__type19,__type32>::type __type33;
    typedef std::integral_constant<long,0> __type35;
    typedef typename cu_basis_funs::type<__type18, __type19, __type26>::__ptype22 __type36;
    typedef indexable_container<__type35, typename std::remove_reference<__type36>::type> __type37;
    typedef typename __combined<__type26,__type37>::type __type38;
    typedef typename cu_basis_funs::type<__type18, __type19, __type38>::__ptype23 __type39;
    typedef indexable_container<__type35, typename std::remove_reference<__type39>::type> __type40;
    typedef typename __combined<__type25,__type37,__type40,__type26>::type __type41;
    typedef __type41 __type42;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type42>::__ptype0 __type43;
    typedef indexable_container<__type2, typename std::remove_reference<__type43>::type> __type44;
    typedef typename __combined<__type14,__type28,__type44>::type __type45;
    typedef typename pythonic::assignable<__type45>::type __type46;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type42>::__ptype1 __type47;
    typedef indexable_container<__type35, typename std::remove_reference<__type47>::type> __type48;
    typedef typename __combined<__type42,__type48>::type __type49;
    typedef typename cu_basis_funs_1st_der::type<__type18, __type33, __type10, __type49>::__ptype2 __type50;
    typedef indexable_container<__type35, typename std::remove_reference<__type50>::type> __type51;
    typedef typename __combined<__type25,__type37,__type40,__type48,__type51,__type26,__type42>::type __type52;
    typedef typename pythonic::assignable<__type52>::type __type53;
    typedef typename pythonic::assignable<__type21>::type __type54;
    typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type55;
    typedef __type55 __type56;
    typedef long __type58;
    typedef decltype(pythonic::operator_::sub(std::declval<__type18>(), std::declval<__type58>())) __type59;
    typedef decltype(pythonic::operator_::add(std::declval<__type59>(), std::declval<__type58>())) __type60;
    typedef decltype(std::declval<__type56>()[std::declval<__type60>()]) __type61;
    typedef __type52 __type62;
    typedef typename std::tuple_element<0,typename std::remove_reference<__type62>::type>::type __type63;
    typedef decltype(pythonic::operator_::mul(std::declval<__type61>(), std::declval<__type63>())) __type64;
    typedef typename std::tuple_element<1,typename std::remove_reference<__type62>::type>::type __type71;
    typedef decltype(pythonic::operator_::mul(std::declval<__type61>(), std::declval<__type71>())) __type72;
    typedef typename std::tuple_element<2,typename std::remove_reference<__type62>::type>::type __type79;
    typedef decltype(pythonic::operator_::mul(std::declval<__type61>(), std::declval<__type79>())) __type80;
    typedef typename std::tuple_element<3,typename std::remove_reference<__type62>::type>::type __type87;
    typedef decltype(pythonic::operator_::mul(std::declval<__type61>(), std::declval<__type87>())) __type88;
    typedef typename __combined<__type54,__type64,__type72,__type80,__type88>::type __type89;
    typedef typename pythonic::assignable<__type89>::type __type90;
    typename pythonic::assignable_noescape<decltype(std::get<1>(knots))>::type dx = std::get<1>(knots);
    __type46 __tuple0 = cu_find_span()(std::get<0>(knots), dx, x);
    typename pythonic::assignable_noescape<decltype(std::get<0>(__tuple0))>::type span = std::get<0>(__tuple0);
    __type53 basis = pythonic::numpy::functor::array{}(pythonic::types::make_tuple(1.85687024e-316, 0.0, 6.93168488263843e-310, 1.31298143e-316), pythonic::numpy::functor::float64{});
    if (pythonic::operator_::eq(der, 0L))
    {
      cu_basis_funs()(span, std::get<1>(__tuple0), basis);
    }
    else
    {
      if (pythonic::operator_::eq(der, 1L))
      {
        cu_basis_funs_1st_der()(span, std::get<1>(__tuple0), dx, basis);
      }
    }
    __type90 y = 0.0;
    y += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 0L)], std::get<0>(basis));
    y += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 1L)], std::get<1>(basis));
    y += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 2L)], std::get<2>(basis));
    y += pythonic::operator_::mul(coeffs[pythonic::operator_::add(pythonic::operator_::sub(span, 3L), 3L)], std::get<3>(basis));
    return y;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_vector::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_vector::operator()(argument_type0&& x, argument_type1&& y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 0L); }, [&] () { return pythonic::operator_::eq(der2, 0L); }))
    {
      cu_eval_spline_2d_vector_00()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
    }
    else
    {
      if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 0L); }, [&] () { return pythonic::operator_::eq(der2, 1L); }))
      {
        cu_eval_spline_2d_vector_01()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
      }
      else
      {
        if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 1L); }, [&] () { return pythonic::operator_::eq(der2, 0L); }))
        {
          cu_eval_spline_2d_vector_10()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
        }
        else
        {
          if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 1L); }, [&] () { return pythonic::operator_::eq(der2, 1L); }))
          {
            cu_eval_spline_2d_vector_11()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 , typename argument_type7 , typename argument_type8 , typename argument_type9 >
  inline
  typename cu_eval_spline_2d_cross::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6, argument_type7, argument_type8, argument_type9>::result_type cu_eval_spline_2d_cross::operator()(argument_type0&& X, argument_type1&& Y, argument_type2&& kts1, argument_type3&& deg1, argument_type4&& kts2, argument_type5&& deg2, argument_type6&& coeffs, argument_type7&& z, argument_type8 der1, argument_type9 der2) const
  {
    if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 0L); }, [&] () { return pythonic::operator_::eq(der2, 0L); }))
    {
      cu_eval_spline_2d_cross_00()(X, Y, kts1, deg1, kts2, deg2, coeffs, z);
    }
    else
    {
      if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 1L); }, [&] () { return pythonic::operator_::eq(der2, 0L); }))
      {
        cu_eval_spline_2d_cross_10()(X, Y, kts1, deg1, kts2, deg2, coeffs, z);
      }
      else
      {
        if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 0L); }, [&] () { return pythonic::operator_::eq(der2, 1L); }))
        {
          cu_eval_spline_2d_cross_01()(X, Y, kts1, deg1, kts2, deg2, coeffs, z);
        }
        else
        {
          if (pythonic::builtins::pythran::and_([&] () { return pythonic::operator_::eq(der1, 1L); }, [&] () { return pythonic::operator_::eq(der2, 1L); }))
          {
            cu_eval_spline_2d_cross_11()(X, Y, kts1, deg1, kts2, deg2, coeffs, z);
          }
        }
      }
    }
    return pythonic::builtins::None;
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_out::cu_eval_spline_2d_vector::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, long>::result_type cu_eval_spline_2d_vector0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& coeffs, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_vector()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_vector::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, long>::result_type cu_eval_spline_2d_vector1(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& coeffs, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_vector()(x, y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_cross::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, long, long>::result_type cu_eval_spline_2d_cross0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& X, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& Y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& coeffs, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_cross()(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_cross::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, long, long>::result_type cu_eval_spline_2d_cross1(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& X, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& Y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& coeffs, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_cross()(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_cross::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, long, long>::result_type cu_eval_spline_2d_cross2(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& X, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& Y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& coeffs, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_cross()(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_cross::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, long, long>::result_type cu_eval_spline_2d_cross3(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& X, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& Y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& coeffs, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& z, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_cross()(X, Y, kts1, deg1, kts2, deg2, coeffs, z, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_scalar::type<double, double, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>, long, long>::result_type cu_eval_spline_2d_scalar0(double&& x, double&& y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>&& coeffs, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_scalar()(x, y, kts1, deg1, kts2, deg2, coeffs, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_2d_scalar::type<double, double, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>, long, long>::result_type cu_eval_spline_2d_scalar1(double&& x, double&& y, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts1, long&& deg1, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& kts2, long&& deg2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>&& coeffs, long&& der1, long&& der2) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_2d_scalar()(x, y, kts1, deg1, kts2, deg2, coeffs, der1, der2);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_1d_vector::type<pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long>::result_type cu_eval_spline_1d_vector0(pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& knots, long&& degree, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& coeffs, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& y, long&& der) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_1d_vector()(x, knots, degree, coeffs, y, der);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_eval_spline_1d_scalar::type<double, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long, pythonic::types::ndarray<double,pythonic::types::pshape<long>>, long>::result_type cu_eval_spline_1d_scalar0(double&& x, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& knots, long&& degree, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& coeffs, long&& der) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_eval_spline_1d_scalar()(x, knots, degree, coeffs, der);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_basis_funs_1st_der::type<long, double, double, pythonic::types::ndarray<double,pythonic::types::pshape<long>>>::result_type cu_basis_funs_1st_der0(long&& span, double&& offset, double&& dx, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& ders) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_basis_funs_1st_der()(span, offset, dx, ders);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_basis_funs::type<long, double, pythonic::types::ndarray<double,pythonic::types::pshape<long>>>::result_type cu_basis_funs0(long&& span, double&& offset, pythonic::types::ndarray<double,pythonic::types::pshape<long>>&& values) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_basis_funs()(span, offset, values);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
inline
typename __pythran_out::cu_find_span::type<double, double, double>::result_type cu_find_span0(double&& xmin, double&& dx, double&& x) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_out::cu_find_span()(xmin, dx, x);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_vector0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"x", "y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_vector0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_vector1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"x", "y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_vector1(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_cross0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"X", "Y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_cross0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_cross1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"X", "Y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_cross1(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_cross2(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"X", "Y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_cross2(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_cross3(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[10+1];
    
    char const* keywords[] = {"X", "Y", "kts1", "deg1", "kts2", "deg2", "coeffs", "z", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8], &args_obj[9]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[7]) && is_convertible<long>(args_obj[8]) && is_convertible<long>(args_obj[9]))
        return to_python(cu_eval_spline_2d_cross3(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[7]), from_python<long>(args_obj[8]), from_python<long>(args_obj[9])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_scalar0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[9+1];
    
    char const* keywords[] = {"x", "y", "kts1", "deg1", "kts2", "deg2", "coeffs", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8]))
        return nullptr;
    if(is_convertible<double>(args_obj[0]) && is_convertible<double>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]) && is_convertible<long>(args_obj[7]) && is_convertible<long>(args_obj[8]))
        return to_python(cu_eval_spline_2d_scalar0(from_python<double>(args_obj[0]), from_python<double>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>(args_obj[6]), from_python<long>(args_obj[7]), from_python<long>(args_obj[8])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_2d_scalar1(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[9+1];
    
    char const* keywords[] = {"x", "y", "kts1", "deg1", "kts2", "deg2", "coeffs", "der1", "der2",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6], &args_obj[7], &args_obj[8]))
        return nullptr;
    if(is_convertible<double>(args_obj[0]) && is_convertible<double>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]) && is_convertible<long>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]) && is_convertible<long>(args_obj[7]) && is_convertible<long>(args_obj[8]))
        return to_python(cu_eval_spline_2d_scalar1(from_python<double>(args_obj[0]), from_python<double>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]), from_python<long>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,pythonic::types::pshape<long,long>>>>(args_obj[6]), from_python<long>(args_obj[7]), from_python<long>(args_obj[8])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_1d_vector0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    
    char const* keywords[] = {"x", "knots", "degree", "coeffs", "y", "der",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<long>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]) && is_convertible<long>(args_obj[5]))
        return to_python(cu_eval_spline_1d_vector0(from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<long>(args_obj[2]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[4]), from_python<long>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_eval_spline_1d_scalar0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    
    char const* keywords[] = {"x", "knots", "degree", "coeffs", "der",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<double>(args_obj[0]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]) && is_convertible<long>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]) && is_convertible<long>(args_obj[4]))
        return to_python(cu_eval_spline_1d_scalar0(from_python<double>(args_obj[0]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[1]), from_python<long>(args_obj[2]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]), from_python<long>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_basis_funs_1st_der0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[4+1];
    
    char const* keywords[] = {"span", "offset", "dx", "ders",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3]))
        return nullptr;
    if(is_convertible<long>(args_obj[0]) && is_convertible<double>(args_obj[1]) && is_convertible<double>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3]))
        return to_python(cu_basis_funs_1st_der0(from_python<long>(args_obj[0]), from_python<double>(args_obj[1]), from_python<double>(args_obj[2]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[3])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_basis_funs0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    
    char const* keywords[] = {"span", "offset", "values",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<long>(args_obj[0]) && is_convertible<double>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2]))
        return to_python(cu_basis_funs0(from_python<long>(args_obj[0]), from_python<double>(args_obj[1]), from_python<pythonic::types::ndarray<double,pythonic::types::pshape<long>>>(args_obj[2])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_cu_find_span0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[3+1];
    
    char const* keywords[] = {"xmin", "dx", "x",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOO",
                                     (char**)keywords , &args_obj[0], &args_obj[1], &args_obj[2]))
        return nullptr;
    if(is_convertible<double>(args_obj[0]) && is_convertible<double>(args_obj[1]) && is_convertible<double>(args_obj[2]))
        return to_python(cu_find_span0(from_python<double>(args_obj[0]), from_python<double>(args_obj[1]), from_python<double>(args_obj[2])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_cu_eval_spline_2d_vector(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_vector0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_vector1(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_eval_spline_2d_vector", "\n""    - cu_eval_spline_2d_vector(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float64[:], int, int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_eval_spline_2d_cross(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_cross0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_cross1(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_cross2(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_cross3(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_eval_spline_2d_cross", "\n""    - cu_eval_spline_2d_cross(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float[:,:], int, int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_eval_spline_2d_scalar(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_scalar0(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_cu_eval_spline_2d_scalar1(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_eval_spline_2d_scalar", "\n""    - cu_eval_spline_2d_scalar(float64, float64, float64[:], int, float64[:], int, float64[:,:], int, int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_eval_spline_1d_vector(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_eval_spline_1d_vector0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_eval_spline_1d_vector", "\n""    - cu_eval_spline_1d_vector(float64[:], float64[:], int, float64[:], float64[:], int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_eval_spline_1d_scalar(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_eval_spline_1d_scalar0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_eval_spline_1d_scalar", "\n""    - cu_eval_spline_1d_scalar(float64, float64[:], int, float64[:], int)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_basis_funs_1st_der(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_basis_funs_1st_der0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_basis_funs_1st_der", "\n""    - cu_basis_funs_1st_der(int, float64, float64, float64[:])", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_basis_funs(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_basis_funs0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_basis_funs", "\n""    - cu_basis_funs(int, float64, float64[:])", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_cu_find_span(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_cu_find_span0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "cu_find_span", "\n""    - cu_find_span(float64, float64, float64)", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "cu_eval_spline_2d_vector",
    (PyCFunction)__pythran_wrapall_cu_eval_spline_2d_vector,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - cu_eval_spline_2d_vector(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float64[:], int, int)"},{
    "cu_eval_spline_2d_cross",
    (PyCFunction)__pythran_wrapall_cu_eval_spline_2d_cross,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - cu_eval_spline_2d_cross(float64[:], float64[:], float64[:], int, float64[:], int, float64[:,:], float[:,:], int, int)"},{
    "cu_eval_spline_2d_scalar",
    (PyCFunction)__pythran_wrapall_cu_eval_spline_2d_scalar,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - cu_eval_spline_2d_scalar(float64, float64, float64[:], int, float64[:], int, float64[:,:], int, int)"},{
    "cu_eval_spline_1d_vector",
    (PyCFunction)__pythran_wrapall_cu_eval_spline_1d_vector,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - cu_eval_spline_1d_vector(float64[:], float64[:], int, float64[:], float64[:], int)"},{
    "cu_eval_spline_1d_scalar",
    (PyCFunction)__pythran_wrapall_cu_eval_spline_1d_scalar,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - cu_eval_spline_1d_scalar(float64, float64[:], int, float64[:], int)"},{
    "cu_basis_funs_1st_der",
    (PyCFunction)__pythran_wrapall_cu_basis_funs_1st_der,
    METH_VARARGS | METH_KEYWORDS,
    "\n""Compute the first derivative of the non-vanishing B-splines\n""at location x, given the knot sequence, polynomial degree\n""and knot span.\n""\n""Supported prototypes:\n""\n""- cu_basis_funs_1st_der(int, float64, float64, float64[:])\n""\n""See function 's_bsplines_non_uniform__eval_deriv' in\n""Selalib's source file\n""'src/splines/sll_m_bsplines_non_uniform.F90'.\n""\n""Parameters\n""----------\n""knots : array_like\n""    Knots sequence.\n""\n""degree : int\n""    Polynomial degree of B-splines.\n""\n""x : float\n""    Evaluation point.\n""\n""span : int\n""    Knot span index.\n""\n""Results\n""-------\n""ders : numpy.ndarray\n""    Derivatives of p+1 non-vanishing B-Splines at location x.\n""\n"""},{
    "cu_basis_funs",
    (PyCFunction)__pythran_wrapall_cu_basis_funs,
    METH_VARARGS | METH_KEYWORDS,
    "\n""Compute the non-vanishing B-splines at location x,\n""given the knot sequence, polynomial degree and knot\n""span.\n""\n""Supported prototypes:\n""\n""- cu_basis_funs(int, float64, float64[:])\n""\n""Parameters\n""----------\n""x : float\n""    Evaluation point.\n""\n""span : int\n""    Knot span index.\n""\n""offset : float\n""\n""Results\n""-------\n""values : numpy.ndarray\n""    Values of p+1 non-vanishing B-Splines at location x.\n""\n""Notes\n""-----\n""The original Algorithm A2.2 in The NURBS Book [1] is here\n""slightly improved by using 'left' and 'right' temporary\n""arrays that are one element shorter.\n""\n"""},{
    "cu_find_span",
    (PyCFunction)__pythran_wrapall_cu_find_span,
    METH_VARARGS | METH_KEYWORDS,
    "\n""Determine the knot span index at location x, given the\n""cell size and start of the domain.\n""\n""Supported prototypes:\n""\n""- cu_find_span(float64, float64, float64)\n""\n""The knot span index i identifies the\n""indices [i-3:i] of all 4 non-zero basis functions at a\n""given location x.\n""\n""Parameters\n""----------\n""xmin : float\n""    The first break point\n""\n""dx : float\n""    The cell size\n""\n""x : float\n""    Location of interest.\n""\n""Returns\n""-------\n""span : int\n""    Knot span index.\n""\n""offset : float\n""\n"""},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "out",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(out)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(out)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("out",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.12.0",
                                      "2023-01-01 17:49:30.697753",
                                      "20e0066da94dafa366f56b9081053e8685a8184401686238278312bbbdb08404");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);


    PYTHRAN_RETURN;
}

#endif