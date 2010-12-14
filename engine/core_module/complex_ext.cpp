/*
   FALCON - The Falcon Programming Language.
   FILE: complex_ext.cpp

   Complex class for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Enrico Lumetti
   Begin: Sat, 05 Sep 2009 21:04:31 +0000

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Complex class for Falcon
   Interface extension functions
*/

/*#
   @beginmodule core
*/

#include "core_module.h"

namespace Falcon {
namespace core {

CoreObject *Complex_Factory( const CoreClass *cls, void *, bool )
{
    return new CoreComplex ( cls );
}

CoreComplex::~CoreComplex ( void )
{ }


bool CoreComplex::hasProperty( const String &key ) const
{
   uint32 pos = 0;
   return generator()->properties().findKey( key, pos );
}


bool CoreComplex::setProperty( const String &key, const Item &item )
{
   if (key == "real")
   {
      m_complex.real( item.forceNumeric() );
      return true;
   }
   if (key == "imag")
   {
      m_complex.imag( item.forceNumeric() );
      return true;
   }

   // found but read only?
   uint32 pos;
   if( generator()->properties().findKey( key, pos ) )
      throw new AccessError( ErrorParam( e_prop_ro, __LINE__ )
            .origin( e_orig_runtime )
            .extra( key ) );

   // no, not found.
   return false;
}

bool CoreComplex::getProperty( const String &key, Item &ret ) const
{
   if (key == "real")
   {
      ret = m_complex.real();
      return true;
   }

   if (key == "imag")
   {
      ret = m_complex.imag();
      return true;
   }

   return defaultProperty( key, ret );
}


FALCON_FUNC Complex_init( ::Falcon::VMachine *vm )
{
   Item *i_real = vm->param(0);
   Item *i_imag = vm->param(1);

   if ( ( i_real != 0 && ! i_real->isOrdinal() )
     || ( i_imag != 0 && ! i_imag->isOrdinal() )
    )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .origin( e_orig_runtime )
          .extra( "[N,N]" ) );
   }

   CoreComplex *c = dyncast<CoreComplex *> ( vm->self().asObject() );
   c->complex() = Complex(
      ( i_real != 0 ) ? i_real->forceNumeric() : 0 ,
      ( i_imag != 0 ) ? i_imag->forceNumeric() : 0 
   );
}

FALCON_FUNC Complex_toString( ::Falcon::VMachine *vm )
{
    CoreComplex *self = dyncast<CoreComplex *>( vm->self().asObject() );
    String res,real,imag;
    Item(self->complex().real() ).toString(real);
    Item(self->complex().imag()).toString(imag);
    res=real+" , "+imag+"i";
    vm->retval(res);
}


FALCON_FUNC Complex_abs( ::Falcon::VMachine *vm )
{
    CoreComplex *self = dyncast<CoreComplex *>( vm->self().asObject() );

    vm->retval( self->complex().abs() );
}

static void s_operands( VMachine* vm, Complex* &one, Complex& two, const CoreClass* &gen )
{
   Item *i_obj = vm->param( 0 );
   bool is_ordinal = i_obj->isOrdinal();

   if ( i_obj == 0 || ! ( i_obj->isOfClass( "Complex" ) || is_ordinal ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "Complex" ) );
   }

   CoreComplex *self = dyncast<CoreComplex *>( vm->self().asObject() );
   if ( is_ordinal )
      two.real( i_obj->forceNumeric() );
   else
      two = (dyncast<CoreComplex *>( i_obj->asObject() ))->complex();
   one = &self->complex();
   gen = self->generator();
}

FALCON_FUNC Complex_add__( ::Falcon::VMachine *vm )
{
   Complex *one, two;
   const CoreClass* gen;
   s_operands( vm, one, two, gen );
   vm->retval( new CoreComplex( (*one) + two, gen ) );
}

FALCON_FUNC Complex_sub__( ::Falcon::VMachine *vm )
{
   Complex *one, two;
   const CoreClass* gen;
   s_operands( vm, one, two, gen );
   vm->retval( new CoreComplex( (*one) - two, gen ) );
}

FALCON_FUNC Complex_mul__( ::Falcon::VMachine *vm )
{
   Complex *one, two;
   const CoreClass* gen;
   s_operands( vm, one, two, gen );
   vm->retval( new CoreComplex( (*one) * two, gen ) );
}

FALCON_FUNC Complex_div__( ::Falcon::VMachine *vm )
{
   Complex *one, two;
   const CoreClass* gen;
   s_operands( vm, one, two, gen );
   vm->retval( new CoreComplex( (*one) / two, gen ) );
}

FALCON_FUNC Complex_compare( ::Falcon::VMachine *vm )
{
   Complex *one, two;
   const CoreClass* gen;
   s_operands( vm, one, two, gen );

   if ( (*one) < two )
      vm->retval( -1 );
   else if ( (*one) > two )
      vm->retval( 1 );
   else
      vm->retval( 0 );
}

FALCON_FUNC Complex_conj( ::Falcon::VMachine *vm )
{
   CoreComplex *self = dyncast<CoreComplex *>( vm->self().asObject() );
   vm->retval( new CoreComplex( self->complex().conj(), self->generator() ) );
}


}
}

/* end of complex_ext.cpp */
