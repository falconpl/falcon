/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: classcontainer.h
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/feathers/containers/classcontainer.cpp"

#include <falcon/pstep.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>
#include <falcon/string.h>
#include <falcon/item.h>

#include "containers_fm.h"
#include "classcontainer.h"
#include "container_mod.h"
#include "iterator_mod.h"
#include "errors.h"

/*#
 @beginmodule containers
*/

namespace Falcon {
using namespace Mod;

namespace {


/** This step checks if an item is in a sequence.
 * Signature: (2: item to be searched) (1: ClassIteartor|iterator) (0: dummy int 1) --> (0: boolean)
 */
class PStepContains: public PStep
{
public:
   PStepContains() {apply = apply_; }
   virtual ~PStepContains(){}
   void describeTo(String& target) const{ target = "PStepContains";}

   static void apply_( const PStep*, VMContext* ctx )
   {
      if(ctx->topData().asInteger() == 0 )
      {
         ctx->popData(2);
         ctx->topData().setBoolean(true);
         ctx->popCode();
         return;
      }
      // remove the previous result
      // -- we still have 1 extra data not removed by op_in
      ctx->popData(1);

      const Item& item = ctx->opcodeParam(1);
      Class* cls = 0;
      void* inst = 0;
      item.forceClassInst(cls,inst);
      Iterator* iter = static_cast<Iterator*>(ctx->topData().asInst());

      long depth = ctx->codeDepth();
      Container* container = iter->container();
      container->lock();
      int32 version = container->version();
      Item current;
      while( iter->next(current, false) )
      {
         fassert( current != 0 );
         container->unlock();

         ctx->pushData(item);
         ctx->pushData(current);
         cls->op_compare(ctx, inst);
         if( ctx->codeDepth() != depth )
         {
            return;
         }

         if(ctx->topData().asInteger() == 0 )
         {
            ctx->popData(2);
            ctx->topData().setBoolean(true);
            ctx->popCode();
            return;
         }

         ctx->popData();
         container->lock();
         if(version != container->version() )
         {
            container->unlock();
            throw FALCON_SIGN_XERROR(ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during contains") );
         }
      }
      container->unlock();
      ctx->popData();
      // add the result
      ctx->topData().setBoolean(false);
      ctx->popCode();
   }
};

//=======================================================================================
// Class Container.
//

/*#
 @class Container
 @brief Base class for all the containers in this module.
 @prop size Count of items stored in this container.
 @prop empty True if the container is empty.

 A container will evaluate to false (in boolean contexts) if it's empty.
*/

/*#
 @method clone Container
 @brief Clone the structure of the container.
 @return A shallow copy of this container.

 The items in the container are not themselves copied.
*/

/*#
 @method iterator Container
 @brief Start iteration from the first element to the last.
 @return An @a Iterator instance pointing to the first item in the container.
*/
FALCON_DECLARE_FUNCTION( iterator, "" )
FALCON_DEFINE_FUNCTION_P1( iterator )
{
   Container* cnt = ctx->tself<Container*>();
   ModuleContainers* mod = static_cast<ModuleContainers*>(methodOf()->module());
   const Class* icls = mod->iteratorClass();
   Iterator* iter = cnt->iterator();
   ctx->returnFrame(FALCON_GC_STORE(icls,iter));
}

/*#
 @method riterator Container
 @brief Start iteration from the last element to the first.
 @return An @a Iterator instance pointing to the last item in the container.
*/

FALCON_DECLARE_FUNCTION( riterator, "" )
FALCON_DEFINE_FUNCTION_P1( riterator )
{
   Container* cnt = ctx->tself<Container*>();
   ModuleContainers* mod = static_cast<ModuleContainers*>(methodOf()->module());
   const Class* icls = mod->iteratorClass();
   Iterator* iter = cnt->riterator();
   ctx->returnFrame(FALCON_GC_STORE(icls,iter));
}

/*#
 @method clean Container
 @brief Removes all the items in the collection.
*/

FALCON_DECLARE_FUNCTION( clean, "" )
FALCON_DEFINE_FUNCTION_P1( clean )
{
   Container* cnt = ctx->tself<Container*>();
   cnt->clear();
   ctx->returnFrame();
}

/*#
 @method contains Container
 @brief Checks if the given item is in the collection.
 @param item The item to be searched
 @return True if the item is in the collection, false otherwise.

 The item is checked for equality via standard equality check,
 which allows specific items to override their equality
 check via their compare method.
*/

FALCON_DECLARE_FUNCTION( contains, "item:X" )
FALCON_DEFINE_FUNCTION_P( contains )
{
   if( pCount == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   Container* cnt = ctx->tself<Container*>();
   const Item& item = *ctx->param(0);
   long depth = ctx->codeDepth();
   if( cnt->contains(ctx, item) )
   {
      ctx->returnFrame(Item().setBoolean(true) );
   }
   else if (ctx->codeDepth() != depth )
   {
      ctx->returnFrame(Item().setBoolean(false));
   }
   // else, don't return frame.
}


static void get_size(const Class*, const String&, void* inst, Item& value )
{
   Container* cnt = static_cast<Container*>(inst);
   value.setInteger(cnt->size());
}


static void get_empty(const Class*, const String&, void* inst, Item& value )
{
   Container* cnt = static_cast<Container*>(inst);
   value.setBoolean(cnt->empty());
}

}

//======================================================================================
// Class.
//

ClassContainerBase::ClassContainerBase( const String &name ):
   Class(name),
   m_stepContains(new PStepContains)
{
   addMethod(new FALCON_FUNCTION_NAME(iterator) );
   addMethod(new FALCON_FUNCTION_NAME(riterator) );
   addMethod(new FALCON_FUNCTION_NAME(clean) );
   addMethod(new FALCON_FUNCTION_NAME(contains) );

   addProperty("size", &get_size );
   addProperty("empty", &get_empty );
}

ClassContainerBase::~ClassContainerBase()
{
   delete m_stepContains;
}

void ClassContainerBase::describe( void* instance, String& target, int depth, int maxlen) const
{
   Container* cont = static_cast<Container*>(instance);
   target += name() + "{";
   if( depth == 0 )
   {
      target +=  "...}";
      return;
   }

   if ( maxlen > 4 )
   {
      maxlen-=4;
   }

   Iterator* iter = cont->iterator();

   int32 version = cont->version();
   bool first = true;
   Item current;
   while( iter->next(current) )
   {
      fassert( current != 0 );

      Class* cls = 0;
      void* data = 0;
      current.forceClassInst(cls, data);
      String temp;
      cls->describe(data, temp, depth-1, maxlen);
      if( ! first )
      {
         target +=", ";
      }
      first = false;
      target += temp;
      if( maxlen > 0 && ((int) target.length()) > maxlen )
      {
         target = target.subString(0, maxlen) + "...}";
         delete iter;
         return;
      }

      if(version != cont->version() )
      {
         target += " <container reset while checking>}";
         delete iter;
         return;
      }
   }

   target += "}";

   delete iter;
}


void ClassContainerBase::dispose( void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   delete cont;
}


void* ClassContainerBase::clone( void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   return cont->clone();
}

void ClassContainerBase::gcMarkInstance( void* instance, uint32 mark ) const
{
   Container* cont = static_cast<Container*>(instance);
   cont->gcMark(mark);
}

bool ClassContainerBase::gcCheckInstance( void* instance, uint32 mark ) const
{
   Container* cont = static_cast<Container*>(instance);
   return cont->currentMark() >= mark;
}

void ClassContainerBase::store( VMContext*, DataWriter*, void* ) const
{
   // do nothing.
}

void ClassContainerBase::restore( VMContext*, DataReader* ) const
{
   // do nothing.
}

void ClassContainerBase::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);

   int32 version = cont->version();
   subItems.reserve(cont->size()+1);
   int64 size = cont->size();
   subItems[0] = size;
   Iterator* iter = cont->iterator();

   Item current;
   for( int64 n = 0; n < size; ++n )
   {
      if( iter->next(current) || version != cont->version() )
      {
         delete iter;
         throw FALCON_SIGN_XERROR(ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during serialization"));
      }

      fassert( current != 0 );
      subItems[0+1].copyFromRemote(current);
   }

   delete iter;
}


void ClassContainerBase::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   int64 size = subItems[0].asInteger();
   for( int64 i = 1; i <= size; ++i )
   {
      cont->append(subItems[i]);
   }
}


void ClassContainerBase::op_isTrue( VMContext* ctx, void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   ctx->topData().setBoolean( ! cont->empty() );
}


void ClassContainerBase::op_in( VMContext* ctx, void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   Item value = ctx->topData();
   ctx->popData(2);
   long depth = ctx->codeDepth();
   if( cont->contains(ctx, value ) )
   {
      ctx->pushData(Item().setBoolean(true));
   }
   else if( ctx->codeDepth() == depth ){
      ctx->pushData(Item().setBoolean(false));
   }
   // else, contains is doing the job right now.
}


void ClassContainerBase::op_iter( VMContext* ctx, void* instance ) const
{
   Container* cont = static_cast<Container*>(instance);
   ModuleContainers* mc = static_cast<ModuleContainers*>(module());
   Iterator* iter = cont->iterator();
   ctx->pushData(FALCON_GC_STORE(mc->iteratorClass(), iter));
}


void ClassContainerBase::op_next( VMContext* ctx, void* ) const
{
   Iterator* iter = static_cast<Iterator*>(ctx->topData().asInst());
   Item value;
   if( ! iter->next(value) ) {
      ctx->pushData(Item().setBreak());
   }
   else {
      ctx->pushData(value);
      if( iter->hasNext() )
      {
         ctx->topData().setDoubt();
      }
   }

}


ClassContainer::ClassContainer():
         ClassContainerBase("Container")
{}

ClassContainer::~ClassContainer()
{}

void* ClassContainer::createInstance() const
{
   // pure virtual at Falcon level
   return 0;
}


}

/* end of classcontainer.cpp */
