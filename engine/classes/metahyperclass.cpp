/*
   FALCON - The Falcon Programming Language.
   FILE: metahyperclass.cpp

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 10 Mar 2012 23:27:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/metahyperclass.cpp"

#include <falcon/classes/metahyperclass.h>

#include <falcon/engine.h>
#include <falcon/hyperclass.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>
#include <falcon/falconclass.h>
#include <falcon/psteps/exprparentship.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/stdhandlers.h>

namespace Falcon
{

MetaHyperClass::MetaHyperClass()
{
   m_name = "$HyperClass";
}

MetaHyperClass::~MetaHyperClass()
{
}


void MetaHyperClass::store( VMContext* , DataWriter* wr, void* instance ) const
{
   HyperClass* hc = static_cast<HyperClass*>( instance );
   
   wr->write( hc->m_name );
   wr->write( hc->m_nParents );
   wr->write( hc->m_ownParentship );
}

void MetaHyperClass::restore( VMContext* ctx, DataReader* dr ) const
{
   String name;
   int nParents;
   bool ownParentship;
   
   dr->read( name );
   dr->read( nParents );
   dr->read( ownParentship );
   
   HyperClass* hc = new HyperClass(name);
   hc->m_nParents = nParents;
   hc->m_ownParentship  = ownParentship;
   
   ctx->pushData( Item( this, hc ) );
}


void MetaHyperClass::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   HyperClass* hc = static_cast<HyperClass*>( instance );
   subItems.reserve(3);
   
   if( hc->m_constructor == 0 )
   {
      subItems.append(Item());
   }
   else {
      subItems.append( Item( hc->m_constructor->handler(), hc->m_constructor ) );
   }
   
   if( hc->m_parentship == 0 )
   {
      subItems.append(Item());
   }
   else {
      subItems.append( Item( hc->m_parentship->handler(), hc->m_parentship ) );
   }
   
   // we know we MUST have a master class, that's the definition of HyperClass.
   subItems.append( Item( hc->m_master->handler(), hc->m_master ) );
}


void MetaHyperClass::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert( subItems.length() == 3 );
   
   HyperClass* hc = static_cast<HyperClass*>( instance );
   Item& iConstuctor = subItems[0];
   Item& iParentship = subItems[1];
   Item& iMaster = subItems[2];
   
   if( iConstuctor.isUser() )
   {
      hc->m_constructor = static_cast<Function*>( iConstuctor.asInst() );
   }
   
   if( iParentship.isUser() ) 
   {
      hc->m_parentship = static_cast<ExprParentship*>( iParentship.asInst() );
   }
   
   hc->m_master = static_cast<FalconClass*>( iMaster.asInst() );
}


const Class* MetaHyperClass::handler() const
{
   static const Class* cls = Engine::handlers()->metaHyperClass();
   return cls;
}

}

/* end of metahyperclass.cpp */
