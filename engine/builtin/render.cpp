/*
   FALCON - The Falcon Programming Language.
   FILE: render.cpp

   Falcon core module -- Render function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/render.cpp"

#include <falcon/builtin/render.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/itemid.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/textwriter.h>
#include <falcon/treestep.h>
#include <falcon/stringstream.h>
#include <falcon/stdhandlers.h>

// These are for TextWriter
#include <falcon/modspace.h>
#include <falcon/module.h>

namespace Falcon {
namespace Ext {

Render::Render():
   Function( "render" )
{
   parseDescription("item:X,stream:[Stream|TextStream]");
}

Render::~Render()
{
}


static void internal_render( Item* elem, TextWriter* tw )
{
   switch( elem->type() )
   {
   case FLC_CLASS_ID_FUNC:
   case FLC_CLASS_ID_CLASS:
      {
         Mantra* func = static_cast<Mantra*>(elem->asInst());
         func->render(tw, 0);
      }
      break;

   case FLC_CLASS_ID_TREESTEP:
      {
         TreeStep* ts = static_cast<TreeStep*>(elem->asInst());
         ts->render(tw, 0);
      }
      break;

   case FLC_CLASS_ID_MODULE:
      {
         Module* cls = static_cast<Module*>(elem->asInst());
         cls->render(tw, 0);
      }
      break;


      // else, falls back to describe.
   default:
      {
         Class* cls = 0;
         void* data = 0;
         elem->forceClassInst(cls, data);
         String target;
         cls->describe(data, target, 1, -1);
         tw->write(target);
      }
      break;
   }

}

void Render::invoke( VMContext* ctx, int32 nParams )
{
   static Class* streamClass = Engine::instance()->handlers()->streamClass();
   static Module* core = ctx->process()->modSpace()->findByName("core");
   static Class* writerClass = core == 0 ? 0 : core->getClass("TextWriter");

   Item *elem;
   Item *i_stream = 0;

   if ( ctx->isMethodic() )
   {
      elem = &ctx->self();
      i_stream = ctx->param(0);
   }
   else
   {
      if( nParams <= 0 )
      {
         throw paramError();
      }

      elem = ctx->params();
      if( nParams > 1 )
      {
         i_stream = elem+1;
      }
   }

   if( i_stream != 0 )
   {
      Class* cls = 0;
      void* inst = 0;
      i_stream->forceClassInst(cls, inst);

      // is this a writer?
      if( writerClass != 0 )
      {
         if( cls->isDerivedFrom(writerClass) )
         {
            // render on the writer.
            TextWriter* tw = static_cast<TextWriter*>(cls->getParentData(writerClass, inst));
            fassert(tw != 0);
            internal_render( elem, tw );
            ctx->returnFrame();
            return;
         }
      }

      if( cls->isDerivedFrom(streamClass) )
      {
         Stream* stream = static_cast<Stream*>(inst);
         LocalRef<TextWriter> twr( new TextWriter(stream) );
         internal_render( elem, &twr );
         ctx->returnFrame();
         return;
      }

      // invalid stream
      throw paramError();
   }

   // if we're here, we don't have a parameter
   StringStream* stream = new StringStream;
   LocalRef<TextWriter> twr( new TextWriter(stream) );
   stream->decref(); // the twr owns the stream
   internal_render( elem, &twr );
   twr->flush();

   String* result = new String;
   stream->closeToString(*result);
   ctx->returnFrame( FALCON_GC_HANDLE(result) );
}

}
}

/* end of render.cpp */
