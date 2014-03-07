/*
   FALCON - The Falcon Programming Language.
   FILE: classdirectory.cpp

   Falcon core module -- Handler to read VFS directory entries.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "modules/native/feathers/vfs/classdirectory.cpp"

#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/uri.h>
#include <falcon/stdhandlers.h>
#include <falcon/classes/classuri.h>
#include <falcon/filestat.h>
#include <falcon/stream.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>

#include "vfs.h"
#include "classdirectory.h"

namespace Falcon {
namespace Ext {


/*#
 @class Directory
 @brief Handler to read VFS directory entries.

 @note This is a pure virtual class; it must be instantiated through
 @a VFS.readDir method.

 @prop uri The original URI of the directory (as a URI entry).
*/

namespace _classDirectory {

/*#
 @method read Directory
 @brief Reads the next entry in the directory.
 @return A string representing a filename in the directory, or nil when done.
 @raise IoError on error while accessing the resource.

 When there are more entries to be read, the method returns with dobut. This means
 that the method can be used as a non-deterministc point in a rule or as a generator
 in a for-in loop.
*/

FALCON_DECLARE_FUNCTION( read ,"" )
void Function_read::invoke( Falcon::VMContext* ctx, int )
{
   Directory* dir = static_cast<Directory*>(ctx->self().asInst());
   String next = dir->next();

   if( next.size() == 0 )
   {
      // close it again, just in case we didn't close it before
      dir->close();
      ctx->returnFrame();
   }
   else
   {
      if( dir->read(dir->next()) )
      {
         // we have 1 more
         ctx->returnFrameDoubt( FALCON_GC_HANDLE(next.clone()) );
      }
      else
      {
         // the last one.
         dir->next().size(0);
         // close it now, in case we never show up again.
         dir->close();
         ctx->returnFrame( FALCON_GC_HANDLE(next.clone()) );
      }
   }
}


/*#
 @method close Directory
 @brief Close the directory handle once it's not used anymore.
 @raise IoError on error while accessing the resource.

 Notice that when the last entry of the directory is read,
 the handler is closed automatically. This method should be invoked
 only when (possibly or actually) not reading all the entries.
*/

FALCON_DECLARE_FUNCTION( close ,"" )
void Function_close::invoke( Falcon::VMContext* ctx, int )
{
   Directory* dir = static_cast<Directory*>(ctx->self().asInst());
   dir->close();
}


/*#
 @method descend Directory
 @brief Invoke a callable item on all the sub-entries of this directory.
 @param func A function to be repeatedly invoked.
 @optparam flat If true, won't descend in sub-directories.
 @raise IoError on error while accessing the resource.

 This method traverses all the entries and descend in all the sub-directories
 of a given directory, invoking a given callback entity on each entry found
 in each directory of the tree.

 The function is called as:

 @code
    func( uri )
 @endcode

 where uri is an URI instance comprising the topmost directory and all the
 directories up to the given file to the current file.

 @note The @b uri entity is overwritten each time; if the method needs to
 store it somewhere, it should create a copy or take its encoding.

 The @b func callback may interrupt further processing by returning explicitly a
 @b false value.
*/

static void descend_next_apply_(const PStep* , VMContext* ctx)
{
   static VFSIface* vfs = &Engine::instance()->vfs();
   // are we asked to be done?
   if( ctx->topData().isBoolean() && ctx->topData().asBoolean() == false )
   {
      ctx->returnFrame();
      return;
   }
   ctx->popData();

   // we're still in our call frame, so...
   if( ctx->topData().asClass() != ctx->self().asClass() )
   {
      throw FALCON_SIGN_XERROR( CodeError, e_stackuf, .extra("Corruption in Directory.descend()") );
   }

   URI* uric = static_cast<URI*>(ctx->local(0)->asInst());

   // get the topmost directory we're working on -- if not nil
   while( ! ctx->topData().isNil() )
   {
      Directory* dir = static_cast<Directory*>(ctx->topData().asInst());
      String fname;
      if( ! dir->read( fname ) )
      {
         ctx->popData(); // and try again.
      }
      else
      {
         // ok, we have a file. Is it a file or a directory?
         uric->path().fulloc(dir->uri().path().encode());
         uric->path().file(fname);
         if( (fname != "." && fname != "..") && vfs->fileType(*uric, true) == FileStat::_dir )
         {
            // then, open it and ready it for next loop
            Directory* dir = vfs->openDir(*uric);
            ctx->pushData( FALCON_GC_STORE( ctx->self().asClass(), dir ) );
         }
         // anyhow, call our callback with our uri
         Item ci = *ctx->local(0);
         Item* callee = ctx->param(0);
         ctx->callerLine(__LINE__+1);
         ctx->callItem( *callee, 1, &ci );
         return;
      }
   }

   // no more Directory* to read
   ctx->returnFrame();
}

FALCON_DECLARE_FUNCTION_EX( descend ,"func:C",
   class NextStep: public PStep
   {
   public:
      NextStep(){ apply= descend_next_apply_; }
      virtual ~NextStep(){}
   };

   NextStep m_next;
   )

void Function_descend::invoke( Falcon::VMContext* ctx, int )
{
   static Class* uriClass = Engine::instance()->stdHandlers()->uriClass();

   Item* i_callable = ctx->param(0);

   if( i_callable == 0 || ! i_callable->isCallable() )
   {
      throw paramError(__LINE__, SRC);
   }


   // prepare the environment.
   Directory* dir = static_cast<Directory*>(ctx->self().asInst());

   // let's see if we really have to do all this.
   if( dir->next().empty() )
   {
      // not really.
      ctx->returnFrame();
      return;
   }

   ctx->addLocals(3);
   // the first var is our (varying) uri.
   URI* uric = new URI;
   *uric = dir->uri();
   *ctx->local(0) = FALCON_GC_STORE( uriClass, uric );

   // we'll use local 1 as a marker that we're done, leaving it nil
   ctx->local(2)->copyInterlocked(ctx->self()); // here we have the topmost entity

   // push the next step
   ctx->pushCode( &m_next );

   // call the method with the first entry
   uric->path().file(dir->next());

   Item ci( uriClass, uric );
   ctx->callerLine(__LINE__+1);
   ctx->callItem( *i_callable, 1, &ci );
}

}


static void get_uri( const Class*, const String&, void *instance, Item& value )
{
   static Class* uriClass = Engine::instance()->stdHandlers()->uriClass();

   Directory* dir = static_cast<Directory*>(instance);
   URI* uric = new URI();
   *uric = dir->uri();
   value = FALCON_GC_STORE(uriClass, uric );
}

//=========================================================
//
//=========================================================


ClassDirectory::ClassDirectory():
         Class("Directory")
{
   addMethod( new _classDirectory::Function_read );
   addMethod( new _classDirectory::Function_close );
   addMethod( new _classDirectory::Function_descend );

   addProperty("uri", &get_uri );
}


ClassDirectory::~ClassDirectory()
{
}

void ClassDirectory::dispose( void* instance ) const
{
   Directory* dir = static_cast<Directory*>(instance);
   delete dir;
}

void* ClassDirectory::clone( void* ) const
{
   return 0;
}

void* ClassDirectory::createInstance() const
{
   return 0;
}

void ClassDirectory::gcMarkInstance( void* instance, uint32 mark ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   fs->gcMark( mark );
}

bool ClassDirectory::gcCheckInstance( void* instance, uint32 mark ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   return fs->currentMark() >= mark;
}


}
}

/* end of classfilestat.cpp */
