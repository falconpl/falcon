/*
   FALCON - The Falcon Programming Language.
   FILE: traits.cpp

   Traits - informations on types for the generic containers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven oct 27 11:02:00 CEST 2006


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/traits.h>
#include <falcon/string.h>

#include <string.h>  // for memset

namespace Falcon {
   
ElementTraits::~ElementTraits()
   {}
   
uint32 VoidpTraits::memSize() const
{
   return sizeof( void * );
}

void VoidpTraits::init( void *targetZone ) const
{
   void **target = (void**) targetZone;
   *target = 0;
}

void VoidpTraits::copy( void *targetZone, const void *sourceZone ) const
{
   void **target = (void**) targetZone;
   *target = (void *) sourceZone;
}

int VoidpTraits::compare( const void *targetZone, const void *sourceZone ) const
{
   void **target = (void**) targetZone;
   if ( ((uint64) *target) < ((uint64)sourceZone) )
      return - 1;
   else if ( ((uint64) *target) > ((uint64)sourceZone) )
      return 1;
   else
      return 0;
}

void VoidpTraits::destroy( void *item ) const
{
   // do nothing
}

bool VoidpTraits::owning() const
{
   return false;
}

uint32 IntTraits::memSize() const
{
   return sizeof( int32 );
}

void IntTraits::init( void *targetZone ) const
{
   int32 *target = (int32*) targetZone;
   *target = 0;
}

void IntTraits::copy( void *targetZone, const void *sourceZone ) const
{
   int32 *target = (int32*) targetZone;
   const int32 *source = (int32*) sourceZone;
   *target = *source;
}

int IntTraits::compare( const void *first, const void *second ) const
{
   const int32 *ifirst = (int32 *) first;
   const int32 *isecond = (int32 *) second;
   if ( *ifirst < *isecond )
      return - 1;
   else if ( *ifirst > *isecond )
      return 1;
   else
      return 0;
}

void IntTraits::destroy( void *item ) const
{
   // do nothing
}

bool IntTraits::owning() const
{
   return false;
}

void StringPtrTraits::init( void *targetZone ) const
{
   String **target = (String**) targetZone;
   *target = 0;
}

uint32 StringPtrTraits::memSize() const
{
   return sizeof( String * );
}

void StringPtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   String **target = (String**) targetZone;
   String *source = (String*) sourceZone;
   *target = source;
}

int StringPtrTraits::compare( const void *first, const void *second ) const
{
   String **ifirst = (String **) first;
   String *isecond = (String *) second;
   return (*ifirst)->compare( *isecond );
}

void StringPtrTraits::destroy( void *item ) const
{
   // do nothing
}

bool StringPtrTraits::owning() const
{
   return false;
}


void StringPtrOwnTraits::destroy( void *item ) const
{
   String **ifirst = (String **) item;
   delete (*ifirst);
}

bool StringPtrOwnTraits::owning() const
{
   return true;
}


uint32 StringTraits::memSize() const
{
   return sizeof( String );
}

void StringTraits::init( void *targetZone ) const
{
   String *target = (String *) targetZone;
   // do minimal initialization
   memset( target, 0, sizeof( String ) ); // all values to zero and false.
   target->manipulator( &csh::handler_static );
}

void StringTraits::copy( void *targetZone, const void *sourceZone ) const
{
   String *target = (String *) targetZone;
   const String *source = (String *) sourceZone;

   // init so that bufferize won't do fancy deletes
   memset( target, 0, sizeof( String ) ); // all values to zero and false.
   target->manipulator( &csh::handler_static );


   // then deep copy the other
   target->bufferize(*source);
}

int StringTraits::compare( const void *first, const void *second ) const
{
   const String *ifirst = (String *) first;
   const String *isecond = (String *) second;
   return ifirst->compare( *isecond );

}

void StringTraits::destroy( void *item ) const
{
   String *ifirst = (String *) item;
   ifirst->manipulator()->destroy( ifirst );
}

bool StringTraits::owning() const
{
   return true;
}


namespace traits {

	 StringTraits* string_dt = 0;
	 VoidpTraits* voidp_dp = 0;
	 IntTraits* int_dp = 0;
	 StringPtrTraits* stringptr_dp = 0;
	 StringPtrOwnTraits* stringptr_own_dp = 0;


      FALCON_DYN_SYM StringTraits &t_string() { if( !string_dt ) string_dt = new StringTraits; return *string_dt; }
      FALCON_DYN_SYM VoidpTraits &t_voidp() { if( !voidp_dp ) voidp_dp = new VoidpTraits; return *voidp_dp; }
      FALCON_DYN_SYM IntTraits &t_int() { if( !int_dp ) int_dp = new IntTraits; return *int_dp; }
      FALCON_DYN_SYM StringPtrTraits &t_stringptr() { if( !stringptr_dp ) stringptr_dp = new StringPtrTraits; return *stringptr_dp; }
      FALCON_DYN_SYM StringPtrOwnTraits &t_stringptr_own() { if( !stringptr_own_dp ) stringptr_own_dp = new StringPtrOwnTraits; return *stringptr_own_dp; }

	  void releaseTraits()
	  {
		  delete string_dt;
		  string_dt = 0;

		  delete voidp_dp;
		  voidp_dp = 0;

		  delete int_dp;
		  int_dp = 0;

		  delete stringptr_dp;
		  stringptr_dp = 0;

		  delete stringptr_own_dp;
		  stringptr_own_dp = 0;

	  }
}

}

/* end of traits.cpp */
