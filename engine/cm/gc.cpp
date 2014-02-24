/*
   FALCON - The Falcon Programming Language.
   FILE: gc.cpp

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/gc.cpp"

#include <falcon/cm/gc.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/function.h>

namespace Falcon {
namespace Ext {


/*#
  @property algorithm GC
  @brief Show or select which automatic collection algorithm is used.

  The algorithm
 */

/*#
  @property limit GC
  @brief Current limit for "green zone" in automatic algorithms.

 */


/*#
  @property baseLimit GC
  @brief Minimum value for the limit property.

 */

namespace CGC {
/*#
 @method perform GC
 @brief Suggests or forces a full garbage collecting.
 @optparam force True to ask for a total garbage collection.
 @optparam wait True to wait until the collection is complete.

 If @b force is false, then the current context only is scheduled for
 inspection as soon as possible. This can cause a delay in the
 execution of subsequent instructions. However, the calling context
 might not see the memory immediately freed, as reclaim happens
 at a later stage.

 If @b force is true, then all the existing contexts are marked
 for inspection, and inspected as soon as possible. If @b wait
 is also true, then the calling context stays blocked until all
 the currently existing contexts are checked, and all the garbage
 memory is actually reclaimed.

 @note If @b force is false, @b wait is ignored. To see memory
 effectively reclaimed in a single agent application after this
 call, set both parameters to true nevertheless.
 */
FALCON_DECLARE_FUNCTION( perform, "force:[B], wait:[B]" );

/*#
 @method suggest GC
 @brief Invites the GC to inspect the oldest or all the contexts
 @optparam all True to ask for inspection on all the contexts.

 The method returns immediately; the GC will try to collect
 the available memory as soon as possible.
 */
FALCON_DECLARE_FUNCTION( suggest, "all:[B]" );

/*#
 @method reset GC
 @brief Clears count of sweep and mark loops.

 After this call, the @a GC.marks and @a GC.sweeps counters
 will be reset to 0.
 */
FALCON_DECLARE_FUNCTION( reset, "all:[B]" );


void Function_perform::invoke( VMContext* ctx, int32 )
{
   static Collector* coll = Engine::instance()->collector();

   Item* i_full = ctx->param(0);
   Item* i_wait = ctx->param(1);

   bool full = i_full != 0 ? i_full->isTrue() : false;
   bool wait = i_wait != 0 ? i_wait->isTrue() : false;

   TRACE( "ClassGC::Method_perform %s, %s", (full? "Full" : "partial"), (wait?"wait": "no wait") );

   if( full )
   {
      if( wait != 0 )
      {
         Shared* sh = new Shared(&ctx->vm()->contextManager());
         coll->performGCOnShared( sh );
         ctx->addWait(sh);
         ctx->engageWait(-1);
      }
      else
      {
         coll->performGC(false);
      }
   }
   else
   {
      ctx->setInspectEvent();
   }

   ctx->returnFrame();
}



void Function_suggest::invoke( VMContext* ctx, int32 )
{
   static Collector* coll = Engine::instance()->collector();
/*
   Item* i_all = ctx->param(0);
   bool all = i_all != 0 ? i_all->isTrue() : false;
   TRACE( "ClassGC::Method_suggest %s", (all? "Full" : "partial") );
*/
   MESSAGE( "ClassGC::Method_suggest");
   coll->suggestGC();
   ctx->returnFrame();
}


void Function_reset::invoke( VMContext* ctx, int32 )
{
   static Collector* coll = Engine::instance()->collector();
   MESSAGE( "ClassGC::Method_reset");

   coll->sweepLoops(true);
   coll->markLoops(true);
   ctx->returnFrame();
}

}

//====================================================
// Properties.
//


static void get_memory( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedMemory();
}


static void get_items( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->storedItems();
}


static void set_enabled( const Class*, const String&, void*, const Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   coll->enable( value.isTrue() );
}

static void get_enabled( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->isEnabled();
}


static void set_status( const Class*, const String&, void*, const Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   int64 v = value.forceInteger();
   if( v < 0 || v > static_cast<int64>(Collector::e_status_red) )
   {
      throw FALCON_SIGN_ERROR( ParamError, e_param_range );
   }
   coll->status( static_cast<Collector::t_status>(v) );
}


static void get_status( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = static_cast<int64>(coll->status());
}


static void set_algorithm( const Class*, const String&, void*, const Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   int32 algo = 0;
   if(!( value.isOrdinal()
            && (algo = static_cast<int32>(value.forceInteger()))
            && algo >= 0 && algo < FALCON_COLLECTOR_ALGORITHM_COUNT)
    ) {

       throw new ParamError( ErrorParam( e_inv_prop_value, __LINE__ , SRC )
          .extra(String("0<=N<").N(FALCON_COLLECTOR_ALGORITHM_COUNT)) );
   }
   coll->setAlgorithm(algo);
}


static void get_algorithm( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = static_cast<int64>(coll->currentAlgorithm());
}


static void set_limit( const Class*, const String&, void*, const Item& value )
{
   static Collector* coll = Engine::instance()->collector();
      if( ! value.isOrdinal() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("N") );
   }

   coll->currentAlgorithmObject()->limit( value.asInteger() );
}


static void get_limit( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->currentAlgorithmObject()->limit();
}


static void set_baseLimit( const Class*, const String&, void*, const Item& value )
{
   static Collector* coll = Engine::instance()->collector();
      if( ! value.isOrdinal() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("N") );
   }

   coll->currentAlgorithmObject()->base( value.forceInteger() );
}


static void get_baseLimit( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->currentAlgorithmObject()->base();
}


static void get_marks( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->markLoops(false);
}


static void get_sweeps( const Class*, const String&, void*, Item& value )
{
   static Collector* coll = Engine::instance()->collector();
   value = coll->sweepLoops(false);
}

//========================================================
//
//========================================================


ClassGC::ClassGC():
   Class("%GC")  
{
   // we don't need an object
   m_bIsFlatInstance = true;

   addProperty( "memory", &get_memory );
   addProperty( "items", &get_items );
   addProperty( "status", &get_status, &set_status );
   addProperty( "enabled", &get_enabled, &set_enabled );

   addProperty( "algorithm", &get_algorithm, &set_algorithm );
   addProperty( "limit", &get_limit, &set_limit );
   addProperty( "baseLimit", &get_baseLimit, &set_baseLimit );
   
   addProperty( "sweeps", &get_sweeps );
   addProperty( "marks", &get_marks );
   
   addConstant( "MANUAL", FALCON_COLLECTOR_ALGORITHM_MANUAL );
   addConstant( "FIXED", FALCON_COLLECTOR_ALGORITHM_FIXED );
   addConstant( "STRICT", FALCON_COLLECTOR_ALGORITHM_STRICT );
   addConstant( "SMOOTH", FALCON_COLLECTOR_ALGORITHM_SMOOTH );
   addConstant( "LOOSE", FALCON_COLLECTOR_ALGORITHM_LOOSE );
   addConstant( "DEFAULT", FALCON_COLLECTOR_ALGORITHM_LOOSE );

   addMethod( new CGC::Function_perform );
   addMethod( new CGC::Function_suggest );
   addMethod( new CGC::Function_reset );
}

ClassGC::~ClassGC()
{}


void* ClassGC::createInstance() const
{
   return 0;
}

void ClassGC::dispose( void* ) const
{
   // nothing to do
}

void* ClassGC::clone( void* instance ) const
{
   return instance;
}


void ClassGC::gcMarkInstance( void*, uint32 ) const
{
   // nothing to do
}

bool ClassGC::gcCheckInstance( void*, uint32 ) const
{
   // nothing to do
   return true;
}

bool ClassGC::op_init( VMContext* , void*, int  ) const
{
   // nothing to do
   return true;
}

}
}

/* end of gc.cpp */
