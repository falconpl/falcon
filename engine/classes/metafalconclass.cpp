/*
   FALCON - The Falcon Programming Language.
   FILE: metafalconclass.cpp

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 10 Mar 2012 23:27:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/metafalconclass.h>
#include <falcon/falconclass.h>
#include <falcon/engine.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/storer.h>

namespace Falcon
{

MetaFalconClass::MetaFalconClass()
{
   m_name = "Class";
}

MetaFalconClass::~MetaFalconClass()
{
}

Class* MetaFalconClass::handler() const
{
   static Class* cls = Engine::instance()->metaFalconClass();   
   return cls;
}

void MetaFalconClass::store( VMContext* ctx, DataWriter* wr, void* inst ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(inst);
   
   wr->write( fcls->name() );
   // if we're storing our own module, then we should save our unconstructed status.
   Storer* storer = ctx->getTopStorer();
   bool isConstructed = storer == 0 || storer->topData() != fcls->module();
   fcls->storeSelf( wr, isConstructed );
}


void MetaFalconClass::restore( VMContext*, DataReader* rd, void*& empty ) const
{
   String name;
   rd->read( name );
   
   FalconClass* fcls = new FalconClass( name );
   
   try {
      fcls->restoreSelf( rd );
      empty = fcls;      
   }
   catch( ... ) {
      delete fcls;
      throw;
   }      
}


void MetaFalconClass::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(instance);
   Storer* storer = ctx->getTopStorer();
   bool isConstructed = storer == 0 || storer->topData() != fcls->module();
   fcls->flattenSelf( subItems, isConstructed );
}
   

void MetaFalconClass::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   FalconClass* fcls = static_cast<FalconClass*>(instance);
   fcls->unflattenSelf( subItems );
}

}

/* end of metafalconclass.cpp */
