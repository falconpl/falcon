/*
   @{MAIN_PRJ}@
   FILE: @{PROJECT_NAME}@_fm.cpp

   @{DESCRIPTION}@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @{AUTHOR}@
   Begin: @{DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{YEAR}@: @{COPYRIGHT}@

   @{LICENSE}@
*/

#define SRC "@{PROJECT_NAME}@"
#include <falcon/falcon.h>
#include "@{PROJECT_NAME}@_fm.h"

namespace Falcon {
namespace Ext {

// declare an anonymous namespace to prevent implicit export.
namespace {

/*# @beginmodule @{PROJECT_NAME}@ */

//===============================================================================
// Functions
//===============================================================================

// The following is a faldoc block for the function
/*--# << remove -- to activate
   @function skeleton
   @brief A basic script function.
   @return Zero.

   This function just illustrates how create a simple Falcon function
   in a module.

   The declaration FALCON_DECLARE_FUNCTION allows to declare the
   "accepted" parameters.

   The real parameter checking is done procedurally inside the function,
   mainly using he macro family FALCON_[N]PCHECK, or manipulating directly
   the nth parameter via ctx->param(nth).
*/

FALCON_DECLARE_FUNCTION(skeleton, "num:N,string:[S]")
FALCON_DEFINE_FUNCTION_P1(skeleton)
{
   int64 number;
   String* string;
   String dfltString("Default");

   if(    ! FALCON_NPCHECK_GET(0,Ordinal,number)
       || ! FALCON_NPCHECK_O_GET(1,String,string, &dfltString)
   )
   {
      // generic parameter error informing the user of the parameter declaration.
      throw paramError(__LINE__, SRC);
   }

   if ( number < 0 )
   {
      // specialized parameter error explaining what's wrong with the parameters.
      throw paramError(__LINE__, SRC, "Number must be > 0");
   }

   // Create a new string we can then send to the garbage collector.
   String* result = new String;
   *result = string->replicate(number);

   // This is the correct way to store a gc-sensible data.
   // -- Notice that the macro FALCON_GC_HANDLE is provided for those entities
   // -- that expose a public const Class* handler() method.
   Item i_result = FALCON_GC_STORE(result->handler(), result);  // === FALCON_GC_HANDLE(result);

   // All the functions should return the frame
   // -- or the the function processing will continue as if it didn't return.
   ctx->returnFrame( i_result );
}


//===============================================================================
// An iterative function using a pstep.
//===============================================================================

FALCON_DECLARE_FUNCTION(skeleton2, "...")
FALCON_DEFINE_FUNCTION_P1(skeleton2)
{
   // We know we're in the right module -- get the pstep from there.
   // It is also possible to declare a PSTEP inside this function class,
   //   but having it in the module is a more flexible solution.
   Module@{MODULE_NAME}@* mod = static_cast<Module@{MODULE_NAME}@*>( this->module() );

   // take the first parameter
   Item* first = ctx->param(0);
   if( first != 0 )
   {
      // invoke our generator step after the class generator step
      ctx->pushCode( mod->stepIterate() );

      // ask the class to perform some job.
      Class* cls;
      void* instance;
      first->forceClassInst(cls, instance);
      ctx->pushData(*first);
      cls->op_toString(ctx, instance);

      // now we abandon the function without leaving the frame;
      // first, the VM will execute the stringify operator of this class,
      // then, it will invoke our separate pstep.
   }
   else {
      // return the frame if we have nothing to do only.
      ctx->process()->textOut()->writeLine("Nothing to do.");
      ctx->returnFrame();
   }
}


class PStepIterate: public PStep
{
public:
   // pstep boot-up: always assign the apply function pointer in constructor.
   PStepIterate() { apply = apply_; }

   // the virtual destructor usually does nothing.
   virtual ~PStepIterate() {}

   // it's a good practice to give infos about what this pstep is, for deep debugging.
   virtual void describeTo(String& target) { target = "Skeleton module PStepIterate"; }

   // this is the pstep workhorse
   static void apply_(const PStep*, VMContext* ctx)
   {
      // get the current parameter count from our sequence ID
      CodeFrame& cf = ctx->currentCode();
      // also, prepare the sequence id for the next call, if there will be any.
      int param = ++cf.m_seqId;
      String prefix = String("Parameter ").N(param).A(": ");
      ctx->process()->textOut()->write(prefix);

      // we are always called with a string in the top stack item
      const Item& topItem = ctx->topData();
      if( topItem.isString() )
      {
         ctx->process()->textOut()->writeLine(*topItem.asString());
      }
      else {
         // the op_toString of the handler class of the last paramter didn't respect the
         // stringification protocol.
         ctx->process()->textOut()->writeLine("????");
      }
      // in every case, we got to remove the top item
      ctx->popData();

      // more paramters to stringify?

      // we're in the frame of the skeleton2 function,
      // so we keep all its parameters
      Item* i_param = ctx->param(param);
      if( i_param == 0 )
      {
         // we're done -- exit from the function.
         ctx->returnFrame();
      }
      else
      {
         // We'll be called again, unless we explicitly call ctx->popCode() or ctx->returnFrame()
         //  -- as our pstep currently lies below the following operation:
         Class* cls;
         void* instance;
         i_param->forceClassInst(cls, instance);
         ctx->pushData(*i_param);
         cls->op_toString(ctx, instance);
      }
   }
};

//===============================================================================
// Classes
//===============================================================================

// This is our structure:
struct Point {
 int x;
 int y;

 // we might want to have some extra space for GC accounting.
 Falcon::uint32 gcMark;
};

// and our handler

/*--# << remove -- to activate
 @class Point
 @param x The x coordinate of the point
 @param y The y coordinate of the point
 @brief A basic script class.

 This function just illustrates how to bind the ineer MOD logic
 with the script. Also, Mod::skeleton(), used by this
 function, is exported through the "service", so it is
 possible to call the MOD logic directly from an embedding
 application, once loaded the module and accessed the service.

 @prop x The x coordinate

*/
class ClassPoint: public Falcon::Class
{
public:
 // Minimal stuff:
 ClassPoint();
 virtual ~ClassPoint();

 // mandatory handlers:
 virtual void dispose( void* instance ) const;
 virtual void* clone( void* instance ) const;
 virtual void* createInstance() const;

 // Relevant Handlers
 virtual void gcMarkInstance( void* instance, Falcon::uint32 mark ) const;
 virtual bool gcCheckInstance( void* instance, Falcon::uint32 mark ) const;

 // Nice to have handlers
 virtual Falcon::int64 occupiedMemory( void* instance ) const;
 virtual void describe( void* instance, Falcon::String& target, int depth = 3, int maxlen = 60 ) const;

 // And... well, you'll want them handlers.
 virtual void store( Falcon::VMContext* ctx, Falcon::DataWriter* stream, void* instance ) const;
 virtual void restore( Falcon::VMContext* ctx, Falcon::DataReader* stream ) const;
};

// constructor of our class.
FALCON_DECLARE_FUNCTION(init, "x:I,y:I")
FALCON_DEFINE_FUNCTION_P1(init)
{
 Falcon::int64 x, y;

 if( ! FALCON_NPCHECK_GET(0, Integer, x) || ! FALCON_NPCHECK_GET(1, Integer, y) )
 {
    throw paramError(__LINE__, SRC);
 }

 // Configure the structure we're given in ctx->self()
 struct Point* pt = ctx->tself<struct Point>();
 pt->x = (int) x;
 pt->y = (int) y;

 // the constructor should always return the self item.
 ctx->returnFrame(ctx->self());
}


/*--# << remove -- to activate
 @method distance Point
 @param pt Another instace of Point
 @return The distance from the given point.
*/
FALCON_DECLARE_FUNCTION(distance, "pt:Point")
FALCON_DEFINE_FUNCTION_P1(distance)
{
 // get somewhere the class we want to check. -- luckily, it's our same class, so...
 static Falcon::Class* pointClass = this->methodOf();
 Falcon::Item* i_pt = ctx->param(0);

 if( i_pt == 0 || ! i_pt->isInstanceOf( pointClass ) )
 {
    throw paramError();
 }

 // we now must be sure to get the correct user data for the point class out of this item.
 struct Point* pt2 = i_pt->castInst<Point>( pointClass );

 // while, we know our self holds a correct pointer
 struct Point* pt1 = ctx->tself<Point>();

 Falcon::numeric d1 = static_cast<Falcon::numeric>(pt1->x - pt2->x);
 Falcon::numeric d2 = static_cast<Falcon::numeric>(pt1->y - pt2->y);

 Falcon::numeric distance = static_cast<Falcon::numeric>(
          sqrt(d1*d1 + d2*d2)
 );

 // flat data do not require GC
 ctx->returnFrame( distance );
}

// now, accessor for properties:
void get_x(const Falcon::Class*, const Falcon::String&, void *instance, Falcon::Item& value )
{
 struct Point* pt = static_cast<struct Point*>(instance);
 value = static_cast<Falcon::int64>(pt->x);
}

void set_x(const Falcon::Class*, const Falcon::String&, void *instance, const Falcon::Item& value )
{
 // we are to check for value to be something sensible.
 if( ! value.isOrdinal() )
 {
    throw FALCON_SIGN_XERROR(Falcon::ParamError, Falcon::e_inv_params, .extra("N") );
 }

 struct Point* pt = static_cast<struct Point*>(instance);
 pt->x = (int) value.asOrdinal();
}

/*--# << remove -- to activate
@property y Point
@brief The y coordinate

The "property" faldoc block allows for a more detailed explanation than the
\@prop command of the \@class block.
*/
void get_y(const Falcon::Class*, const Falcon::String&, void *instance, Falcon::Item& value )
{
 struct Point* pt = static_cast<struct Point*>(instance);
 value = static_cast<Falcon::int64>(pt->y);
}

void set_y(const Falcon::Class*, const Falcon::String&, void *instance, const Falcon::Item& value )
{
 // we are to check for value to be something sensible.
 if( ! value.isOrdinal() )
 {
    throw FALCON_SIGN_XERROR(Falcon::ParamError, Falcon::e_inv_params, .extra("N") );
 }

 struct Point* pt = static_cast<struct Point*>(instance);
 pt->y = (int) value.asOrdinal();
}

// Let's define our class:
ClassPoint::ClassPoint():
       Class("Point")
{
 // the constructor
 setConstuctor( new FALCON_FUNCTION_NAME(init) );

 // a method
 addMethod( new FALCON_FUNCTION_NAME(distance) );

 // and properties:
 addProperty("x", &get_x, &set_x);
 addProperty("y", &get_y, &set_y);

}

ClassPoint::~ClassPoint()
{
 // nothing to delete in class
}

// This method is called when the GC thinks an object should be destroyed.
void ClassPoint::dispose( void* instance ) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 delete pt;
}

// This method is called when the user asks ofr an explicit cloning.
void* ClassPoint::clone( void* instance ) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 struct Point* copy = new struct Point;

 copy->x = pt->x;
 copy->y = pt->y;
 return copy;
}

// This is invoked prior the constructor receives an unconfigured <self> object.
void* ClassPoint::createInstance() const
{
 struct Point* pt = new struct Point;
 pt->x = 0;
 pt->y = 0;
 return pt;

 // you can create the item directly in the constructor, if you wish, using:
 // return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassPoint::gcMarkInstance( void* instance, Falcon::uint32 mark ) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 pt->gcMark = mark;
}

// this tells the GC if an object is alive or not.
bool ClassPoint::gcCheckInstance( void* instance, Falcon::uint32 mark ) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 return pt->gcMark >= mark;
}


Falcon::int64 ClassPoint::occupiedMemory( void* ) const
{
 // return the estemed size of the object.
 // We add 16 to account for extra memory blocks used by new allocator.
 return (Falcon::int64) sizeof(struct Point) + 16;
}

void ClassPoint::describe( void* instance, Falcon::String& target, int, int) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 // we want to render this object to string as "Point (x,y)"
 target = Falcon::String("Point (").N(pt->x).A(",").N(pt->y).A(")");
}

// basic serialization step:
void ClassPoint::store( Falcon::VMContext*, Falcon::DataWriter* stream, void* instance ) const
{
 struct Point* pt = static_cast<struct Point*>(instance);
 // the static cast is a formality to avoid data mismatch on old compilers where int might not be 32 bit.
 stream->write(static_cast<Falcon::int32>(pt->x));
 stream->write(static_cast<Falcon::int32>(pt->y));
}

void ClassPoint::restore( Falcon::VMContext* ctx, Falcon::DataReader* stream ) const
{
 // the de-serialization requires us to put a fully configured item on top of the context,
 // We also need to store the item in the GC.

 Falcon::int32 x, y;
 // first we shall read the components of the object,
 // so, in case of throws from I/O, we won't leak.

 stream->read(x);
 stream->read(y);

 struct Point* pt = new struct Point;
 pt->x = x;
 pt->y = y;

 // time to give our object back to the engine.
 // As THIS is the class handling the object...
 ctx->pushData( FALCON_GC_STORE(this, pt) );
}
}

//===========================================================
// The module
//===========================================================

Module@{MODULE_NAME}@::Module@{MODULE_NAME}@():
         Module("@{PROJECT_NAME}@")
{
   // create here a pstep that might theoretically be used
   // by all the entities of the module.
   m_stepIterate = new PStepIterate;

   // equivalent to calls to addMantra()
   *this
      << new FALCON_FUNCTION_NAME(skeleton)
      << new FALCON_FUNCTION_NAME(skeleton2)
      << new ClassPoint
   ;
}

Module@{MODULE_NAME}@::~Module@{MODULE_NAME}@()
{
   delete m_stepIterate;
   // Classes and functions that are added to the module
   // are automatically deleted.
}


}} // namespace Falcon::Ext

/* end of @{PROJECT_NAME}@_fm.cpp */
