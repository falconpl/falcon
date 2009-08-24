/*
   FALCON - The Falcon Programming Language.
   FILE: traits.h

   Traits - informations on types for the generic containers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven oct 27 11:02:00 CEST 2006


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef fal_traits_h
#define fal_traits_h

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class FALCON_DYN_CLASS ElementTraits
{
public:
   virtual ~ElementTraits();
   virtual uint32 memSize() const = 0;
   virtual void init( void *itemZone ) const = 0;
   virtual void copy( void *targetZone, const void *sourceZone ) const = 0;
   virtual int compare( const void *first, const void *second ) const = 0;
   virtual void destroy( void *item ) const = 0;
   virtual bool owning() const = 0;
};

class FALCON_DYN_CLASS VoidpTraits: public ElementTraits
{
public:
   virtual uint32 memSize() const;
   virtual void init( void *itemZone ) const;
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

class FALCON_DYN_CLASS IntTraits: public ElementTraits
{
public:
   virtual uint32 memSize() const;
   virtual void init( void *itemZone ) const;
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

class FALCON_DYN_CLASS StringPtrTraits: public ElementTraits
{
public:
   virtual uint32 memSize() const;
   virtual void init( void *itemZone ) const;
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

class FALCON_DYN_CLASS StringPtrOwnTraits: public StringPtrTraits
{
public:
   virtual ~StringPtrOwnTraits() {}
   virtual bool owning() const;
   virtual void destroy( void *item ) const;
};

class FALCON_DYN_CLASS StringTraits: public ElementTraits
{
public:
   virtual uint32 memSize() const;
   virtual void init( void *itemZone ) const;
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};


namespace traits {
      extern FALCON_DYN_SYM StringTraits &t_string();
      extern FALCON_DYN_SYM VoidpTraits &t_voidp();
      extern FALCON_DYN_SYM IntTraits &t_int();
      extern FALCON_DYN_SYM StringPtrTraits &t_stringptr();
      extern FALCON_DYN_SYM StringPtrOwnTraits &t_stringptr_own();
	  void FALCON_DYN_SYM releaseTraits();
}

}

#endif

/* end of traits.h */
