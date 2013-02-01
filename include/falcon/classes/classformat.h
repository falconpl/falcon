/*
   FALCON - The Falcon Programming Language.
   FILE: classformat.h

   Format type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Feb 2013 23:31:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSFORMAT_H_
#define _FALCON_CLASSFORMAT_H_

#include <falcon/setup.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/string.h>

#include <falcon/pstep.h>
namespace Falcon
{

/**
 Class explicitly holding a format.
 */

class FALCON_DYN_CLASS ClassFormat: public ClassUser
{
public:

   ClassFormat();
   virtual ~ClassFormat();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* dw, void* data ) const;
   virtual void restore( VMContext* , DataReader* dr ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   virtual bool op_init( VMContext* ctx, void*, int32 pcount ) const;
   virtual void op_compare( VMContext* ctx, void* self ) const;

private:

   //====================================================
   // Properties.
   //
   FALCON_DECLARE_PROPERTY( original );

   FALCON_DECLARE_METHOD( format, "X" );
   FALCON_DECLARE_METHOD( parse, "S" );

   /*
   FALCON_DECLARE_PROPERTY( isText );

   FALCON_DECLARE_METHOD( cmpi, "S" );
   FALCON_DECLARE_METHOD( endsWith, "S" );
   FALCON_DECLARE_METHOD( fill, "S" );
   FALCON_DECLARE_METHOD( join, "..." );
   FALCON_DECLARE_METHOD( merge, "A" );
   FALCON_DECLARE_METHOD( replace, "S,S" );
   FALCON_DECLARE_METHOD( replicate, "N" );
   FALCON_DECLARE_METHOD( rfind, "S" );
   FALCON_DECLARE_METHOD( rsplit, "S" );
   FALCON_DECLARE_METHOD( splittr, "S" );
   FALCON_DECLARE_METHOD( startsWith, "S" );
   FALCON_DECLARE_METHOD( wmatch, "S" );
   */
};

}

#endif /* _FALCON_CLASSFORMAT_H_ */

/* end of classformat.h */
