/*
   FALCON - The Falcon Programming Language.
   FILE: class.cpp

   Class definition of a Falcon Class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 15:01:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/class.cpp"

#include <falcon/trace.h>
#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/itemid.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>
#include <falcon/bom.h>
#include <falcon/error.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>
#include <falcon/extfunc.h>
#include <falcon/textwriter.h>
#include <falcon/symbol.h>

#include <falcon/ov_names.h>

#include <falcon/function.h>
#include <falcon/extfunc.h>

#include <map>

namespace Falcon {

static void setReadOnly( const Class*, const String& name, void *, const Item& )
{
   throw new AccessError( ErrorParam( e_prop_ro, __LINE__, SRC ).extra(name) );
}

static void getWriteOnly( const Class*, const String& name, void *, Item& )
{
   throw new AccessError( ErrorParam( e_prop_wo, __LINE__, SRC ).extra(name) );
}

class Property
{
public:
   Class::setter setFunc;
   Class::getter getFunc;

   Function* method;
   Item value;
   bool bHidden;
   bool bStatic;
   bool bConst;
   Property() {}

   Property( Class::getter getf, Class::setter setf, bool s, bool h, bool c, Function* meth, Item& v )
   {
      set( getf, setf, s, h, c, meth, v );
   }
   ~Property() {}

   void set( Class::getter getf, Class::setter setf, bool s, bool h, bool c, Function* meth, const Item& v )
   {
      if ( getf == 0 )
      {
         h = true;
         getf = &getWriteOnly;
      }

      getFunc = getf;
      setFunc = setf == 0 ? setReadOnly : setf;

      bStatic = s;
      bHidden = h;
      bConst = c;
      method = meth;
      value = v;
   }
};


class Class::Private
{
public:
   typedef std::map<String, Property> PropertyMap;
   PropertyMap m_props;

   Function* m_constructor;

   Private():
      m_constructor(0)
   {}

   ~Private() {
   }
};


Class::Class( const String& name ):
   Mantra( name, 0, 0, 0 ),
   m_bIsFalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_bHasSharedInstances(false),
   m_userFlags(0),
   m_typeID( FLC_ITEM_USER ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
   _p = new Private;
   m_parent = 0;
}


Class::Class( const String& name, int64 tid ):
   Mantra( name, 0, 0, 0 ),
   m_bIsFalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_bHasSharedInstances(false),
   m_userFlags(0),
   m_typeID( tid ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
   _p = new Private;
   m_parent = 0;
}


Class::Class( const String& name, Module* module, int line, int chr ):
   Mantra( name, module, line, chr ),
   m_bIsFalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( FLC_ITEM_USER ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
   _p = new Private;
   m_parent = 0;
}


Class::Class( const String& name, int64 tid, Module* module, int line, int chr ):
   Mantra( name, module, line, chr ),
   m_bIsFalconClass( false ),
   m_bIsErrorClass( false ),
   m_bIsFlatInstance(false),
   m_userFlags(0),
   m_typeID( tid ),
   m_clearPriority( 0 )
{
   m_category = e_c_class;
   _p = new Private;
   m_parent = 0;
}


Class::~Class()
{
   TRACE1( "Destroying class %s.%s",
      m_module != 0 ? m_module->name().c_ize() : "<internal>",
      m_name.c_ize() );

   Private::PropertyMap::iterator iter = _p->m_props.begin();
   while( _p->m_props.end() != iter )
   {
      Property& prop = iter->second;
      if( prop.method != 0 && prop.method->methodOf() == this )
      {
         delete prop.method;
      }
      ++iter;
   }
   delete _p;
}


Class* Class::handler() const
{
   static Class* meta = Engine::handlers()->metaClass();
   return meta;
}

const Class* Class::getParent( const String& name ) const
{
   if( m_parent == 0 || m_parent->name() != name ) {
      return 0;
   }
   return m_parent;
}


void Class::render( TextWriter* tw, int32 depth ) const
{
   tw->write(PStep::renderPrefix(depth));

   tw->write( "/* Native class " );
   tw->write( this->fullName() );
   tw->write( " */" );

   if( depth >= 0 )
   {
      tw->write("\n");
   }
}


bool Class::isDerivedFrom( const Class* cls ) const
{
   return this == cls ||
            (
                     m_parent != 0
                     && m_parent->isDerivedFrom(cls));
}


void Class::enumerateParents( Class::ClassEnumerator& pe ) const
{
   if( m_parent != 0 ) { pe(m_parent); }
}

void* Class::getParentData( const Class* parent, void* data ) const
{
   if( parent == this ) return data;
   if( m_parent != 0 ) return m_parent->getParentData( parent, data );
   return 0;
}

int64 Class::occupiedMemory( void* ) const
{
   return 0;
}
 
void Class::store( VMContext*, DataWriter*, void* ) const
{
      throw new UnserializableError(ErrorParam( e_unserializable, __LINE__, __FILE__ )
      .origin( ErrorParam::e_orig_vm )
      .extra(name() + " doesn't support store"));
}


void Class::restore( VMContext*, DataReader*) const
{
   throw new UnserializableError(ErrorParam( e_unserializable, __LINE__, __FILE__ )
      .origin( ErrorParam::e_orig_vm )
      .extra(name() + "  doesn't support restore"));
}


void Class::flatten( VMContext*, ItemArray&, void* ) const
{
   // normally does nothing
}


void Class::unflatten( VMContext*, ItemArray&, void* ) const
{
   // normally does nothing
}


void Class::gcMarkInstance( void*, uint32 ) const
{
   // normally does nothing
}


bool Class::gcCheckInstance( void*, uint32 ) const
{
   return true;
}


void Class::describe( void* instance, String& target, int depth, int maxlen) const
{
   String temp;
   
   target.reserve(128);
   target.size(0);
   
   target.append(name());

   if( depth == 0 )
   {
       target += "{...}";
   }
   else
   {
      Private::PropertyMap::const_iterator iter = _p->m_props.begin();
      bool bFirst = true;
      
      target += '{';
      
      while( iter != _p->m_props.end() )
      {
         const Property* prop = &iter->second;
         if( ! (prop->bHidden || prop->bStatic) )
         {
            Item value;
            prop->getFunc( this, iter->first, instance, value );

            if( ! (value.isFunction() || value.isMethod()) )
            {
               if( bFirst )
               {
                  bFirst = false;
               }
               else
               {
                  target += ','; target += ' ';
               }

               value.describe( temp, depth-1, maxlen );
               target.append( iter->first );
               target.append('=');
               target.append(temp);
               temp.size(0);
            }
         }
         
         ++iter;
      }
            
      target += '}';
   }
}


void Class::inspect( void* instance, String& target, int depth ) const
{
   String temp;

   target.reserve(128);
   target.size(0);

   target.append(name());

   if( depth == 0 )
   {
       target += "{...}";
   }
   else
   {
      Private::PropertyMap::const_iterator iter = _p->m_props.begin();

      target += "{";

      while( iter != _p->m_props.end() )
      {
         const String& name = iter->first;
         const Property* prop = &iter->second;
         target.append("\n");
         target.append( name );

         if( prop->bHidden )
         {
            target.append( " (hidden)");
         }
         else {
            Item value;
            prop->getFunc( this, iter->first, instance, value );

            if( value.isFunction() )
            {
               target += "(";
               target += value.asFunction()->signature();
               target += ")";
            }
            else if( value.isMethod() )
            {
               target += "(";
               target += value.asMethodFunction()->signature();
               target += ")";
            }
            else
            {
               Class* cls = 0;
               void* inst = 0;
               value.forceClassInst(cls, inst);
               cls->inspect( inst, temp, depth-1 );
               target.append('=');
               target.append(temp);
               temp.size(0);
            }
         }

         ++iter;
      }

      if( ! _p->m_props.empty() )
      {
         target += '\n';
      }

      target += '}';
   }
}

void Class::enumerateProperties( void*, Class::PropertyEnumerator& pe ) const
{
   Private::PropertyMap::iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      pe( iter->first );
      ++iter;
   }
}


void Class::enumeratePV( void* inst, Class::PVEnumerator& pve) const
{
   Private::PropertyMap::iterator iter = _p->m_props.begin();
   while( iter != _p->m_props.end() )
   {
      Property& prop = iter->second;
      if( prop.method != 0 )
      {
         Item temp;
         temp.setFunction(prop.method);
         pve(iter->first, temp);
      }
      else if( prop.bConst )
      {
         pve( iter->first, prop.value );
      }
      else if( prop.getFunc != 0 ) 
      {
         Item temp;
         prop.getFunc(this, iter->first, inst, temp );
         pve( iter->first, temp );
      }
      ++iter;
   }
}


bool Class::hasProperty( void*, const String& prop ) const
{
   Private::PropertyMap::iterator iter = _p->m_props.find( prop );
   return iter != _p->m_props.end();
}


void Class::enumerateSummonings( void* instance, PropertyEnumerator& cb ) const
{
   enumerateProperties( instance, cb );
}

//==========================================================
// Property management
//==========================================================

void Class::addProperty( const String& name, getter get, setter set, bool isStatic, bool isHidden )
{
   _p->m_props[name].set(get, set, isStatic, isHidden, false, 0, Item() );
}


void Class::addMethod( Function* func, bool isStatic )
{
   Private::PropertyMap::iterator pos = _p->m_props.find( func->name() ); 
   
   if( pos != _p->m_props.end() )
   {
      Property& prop = pos->second;
      if( prop.method != 0 && prop.method->methodOf() == this )
      {
         delete prop.method;
      }

      prop.set( 0, 0, isStatic, true, true, func, Item() );
   }
   else {
      _p->m_props[func->name()].set(0, 0, isStatic, true, true, func, Item() );
   }

   if( func->methodOf() == 0 )
   {
      func->methodOf(this);
   }
}


Function* Class::addMethod( const String& name, ext_func_t func, const String& prototype, bool isStatic )
{
   Function* f = new ExtFunc(name, func, 0, 0);
   f->parseDescription( prototype );
   addMethod( f, isStatic );
   return f;
}

void Class::setConstuctor( Function* func )
{
   delete _p->m_constructor;
   _p->m_constructor = func;
   func->methodOf( this );   
   func->name("init");
}

 
void Class::setConstuctor( ext_func_t func, const String& prototype )
{
   Function* f = new ExtFunc("init", func, 0, 0);
   f->parseDescription( prototype );
   setConstuctor( f );
}


Function* Class::getConstructor() const
{
   return _p->m_constructor;
}


void Class::addConstant( const String& name, const Item& value )
{
   _p->m_props[name].set(0, 0, true, true, true, 0, value );
}


void Class::setParent( const Class* parent )
{
   m_parent = parent;
   // copy all the properties here.
   Private::PropertyMap& props = parent->_p->m_props;
   Private::PropertyMap::iterator iter = props.begin();
   while( iter != props.end() )
   {
      Property& prop = iter->second;
      // do not copy sub-classes
      if( parent->m_parent == 0 || prop.value.asInst() != parent->m_parent )
      {
         _p->m_props[iter->first] = prop; // use default copy
      }
      ++iter;
   }

   // save also the base class as property.
   _p->m_props[parent->name()].set(0,0,true,true,true,0, Item(parent->handler(), const_cast<Class*>(parent)));
}

void Class::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      Mantra::gcMark(mark);

      // copy all the properties here.
      Private::PropertyMap& props = _p->m_props;
      Private::PropertyMap::iterator iter = props.begin();
      while( iter != props.end() )
      {
         Property& prop = iter->second;
         prop.value.gcMark(mark); // this will include our mehtods.
         ++iter;
      }

      if( _p->m_constructor != 0 )
      {
         _p->m_constructor->gcMark(mark);
      }
   }
}


void Class::op_compare( VMContext* ctx, void* self ) const
{
   void* inst;
   Item *op1, *op2;
   
   ctx->operands( op1, op2 );
   
   if( op2->isUser() )
   {
      if( (inst = op2->asInst()) == self )
      {
         ctx->stackResult(2, 0 );
         return;
      }

      byte* bself = static_cast<byte*>(self);
      byte* bop2 = static_cast<byte*>(op2->asInst());

      ctx->stackResult(2, (int64)  (bself - bop2) );
      return;
   }
   else if( op2->type() == op1->type() )
   {
      switch(op2->type())
      {
      case FLC_ITEM_NIL: ctx->stackResult(2, (int64) 0 ); return;
      case FLC_ITEM_BOOL: ctx->stackResult(2, (int64) (op2->asBoolean() == op1->asBoolean() ? 0 : (op2->asBoolean() ? 1: -1))); return;
      case FLC_ITEM_INT: ctx->stackResult(2, (int64) op2->asInteger() - op1->asInteger()); return;
      case FLC_ITEM_NUM: ctx->stackResult(2, op2->asNumeric() < op1->asNumeric() ? -1 : op2->asNumeric() > op1->asNumeric() ? 1 : 0); return;
      case FLC_ITEM_METHOD: ctx->stackResult(2, (int64)(op2->asMethodFunction() - op2->asMethodFunction())); return;
      }
   }

   // we have no information about what an item might be here, but we can
   // order the items by type
   ctx->stackResult(2, (int64) op2->type() - op1->type() );
}


void Class::onInheritanceResolved( ExprInherit* )
{
   // do nothing
}

void Class::delegate( void*, Item*, const String& ) const
{
   throw FALCON_SIGN_ERROR( AccessError, e_non_delegable );
}


Selectable* Class::getSelectableInterface( void* ) const
{
   return 0;
}
//=====================================================================
// VM Operator override.
//

bool Class::op_init( VMContext* ctx, void* inst, int32 pCount ) const
{
   if( _p->m_constructor != 0 )
   {
      ctx->callInternal( _p->m_constructor, pCount, Item(this,inst) );
      // the method either went deep, or returned self.
      // in either cases, the metaclass must do nothing.
      return true;
   }

   throw FALCON_SIGN_XERROR( OperandError, e_invop, .extra("init") );
}


void Class::op_neg( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("neg") );
}

void Class::op_add( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("add") );
}

void Class::op_sub( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("sub") );
}


void Class::op_mul( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("mul") );
}


void Class::op_div( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("div") );
}


void Class::op_mod( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("mod") );
}


void Class::op_pow( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("pow") );
}

void Class::op_shr( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("shift right (>>)") );
}

void Class::op_shl( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
            .extra("shift left (<<)") );
}


void Class::op_aadd( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto add (+=)") );
}


void Class::op_asub( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto sub (-=)") );
}


void Class::op_amul( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto amul (-=)") );;
}


void Class::op_adiv( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto div (/=)") );
}


void Class::op_amod( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto mod (%=)") );
}


void Class::op_apow( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto pow (**=)") );
}

void Class::op_ashr( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto shr (>>=)") );
}

void Class::op_ashl( VMContext* ctx, void*  ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Auto shl (<<=)") );
}


void Class::op_inc( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Pre-inc (++x)") );
}


void Class::op_dec( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Pre-dec (--x)") );
}


void Class::op_incpost( VMContext* ctx, void*) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Post-inc (x++)") );
}


void Class::op_decpost( VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Post-dec (x--)") );
}


void Class::op_call( VMContext* ctx, int32 count , void* ) const
{
   ctx->popData(count);
}


void Class::op_getIndex(VMContext* ctx, void* ) const
{
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
               .extra("Get index []") );
}


void Class::op_setIndex(VMContext* ctx, void* ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   FALCON_RESIGN_XERROR( OperandError, e_invop, ctx,
                .extra("Set index []=") );
}


void Class::op_getProperty( VMContext* ctx, void* data, const String& propName ) const
{
   static BOM* bom = Engine::instance()->getBom();

   Private::PropertyMap::iterator iter = _p->m_props.find( propName );
   if( iter != _p->m_props.end() )
   {
      Property& prop = iter->second;
      if( prop.method != 0 ) 
      {
         ctx->topData().setUser(ctx->topData().asClass(), data);
         ctx->topData().methodize(prop.method);
      }
      else if (prop.bConst )
      {
         ctx->topData() = prop.value;
      }
      else
      {
         prop.getFunc( ctx->topData().asClass(), iter->first, data, ctx->topData() );
      }

      return;
   }

   // try to find a valid BOM propery.
   BOM::handler handler = bom->get( propName );
   if ( handler != 0  )
   {
      handler( ctx, this, data );
   }
   else
   {
      FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                   .extra(propName) );
   }
}


void Class::op_setProperty( VMContext* ctx, void* data, const String& prop ) const
{
   Private::PropertyMap::iterator iter = _p->m_props.find( prop );
   if( iter != _p->m_props.end() )
   {
      Property& prop = iter->second;
      if( !prop.bConst )
      {
         ctx->popData();
         prop.setFunc( this, iter->first, data, ctx->topData() );
      }
      return;
   }
   FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                   .extra(prop) );
}


static void internal_callprop( const Class* cls, VMContext* ctx, void* instance, const String& message, int32 pCount, Property& prop )
{
   ctx->popData();
   if( prop.method != 0 )
   {
      ctx->callInternal(prop.method, pCount, ctx->opcodeParam(pCount));
      return;
   }
   else {
      if( pCount > 0 ) {

         if( ! prop.bConst )
         {
            ctx->popData( pCount-1 );
            prop.setFunc(cls, message, instance, ctx->topData() );
            return;
         }
         else {
            FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                               .extra(message) );
            return;
         }
      }
      else {
         prop.getFunc( cls, message, instance, ctx->addDataSlot() );
         return;
      }
   }
}

void Class::op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool isOptional ) const
{
   static BOM* bom = Engine::instance()->getBom();

   Private::PropertyMap::iterator iter = _p->m_props.find( message );
   if( iter != _p->m_props.end() )
   {
      Property& prop = iter->second;
      internal_callprop( this, ctx, instance, message, pCount, prop );
      return;
   }

   if ( message == "respondsTo" ) {
      Item* msg;
      if( pCount < 1 || ! (( msg = &ctx->opcodeParam(pCount-1) )->isString() || msg->isSymbol()) ) {
         FALCON_RESIGN_XERROR( ParamError, e_inv_params, ctx, .extra("S|Symbol") );
         return;
      }

      bool res = hasProperty(instance, msg->isString() ? *msg->asString() : msg->asSymbol()->name() );
      ctx->stackResult(pCount+1, Item().setBoolean(res) );
      return;
   }

   if( message == "summon" )
   {
      Item* msg;
      if( pCount < 1 || ! (( msg = &ctx->opcodeParam(pCount-1) )->isString() || msg->isSymbol()) ) {
         FALCON_RESIGN_XERROR( ParamError, e_inv_params, ctx, .extra("S|Symbol") );
         return;
      }
      pCount--;
      String msgName = msg->isString() ? *msg->asString() : msg->asSymbol()->name();
      ctx->removeData(pCount,1);
      op_summon( ctx, instance, msgName, pCount, isOptional );
      return;
   }

   if( message == "vsummon" )
   {
      Item* i_msg, *i_params;
      if( pCount < 2
               || ! (( i_msg = &ctx->opcodeParam(pCount-1) )->isString() || i_msg->isSymbol())
               || ! ( i_params = &ctx->opcodeParam(pCount-2) )->isArray() ) {
         FALCON_RESIGN_XERROR( ParamError, e_inv_params, ctx, .extra("S,A") );
         return;
      }
      String msgName = i_msg->isString() ? *i_msg->asString() : i_msg->asSymbol()->name();
      ItemArray* params = i_params->asArray();
      ctx->removeData(pCount-1,pCount);
      for( length_t i = 0; i < params->length(); i++ )
      {
         ctx->pushData(params->at(i));
      }

      op_summon( ctx, instance, msgName, params->length(), isOptional );
      return;
   }

   if ( message == "delegate" ) {
      if( pCount == 0 ) {
         delegate( instance, 0, "*" );
      }
      else if( pCount == 1 ) {
         delegate( instance, &ctx->opcodeParam(0), "*" );
      }
      else {
         Item* delegated = &ctx->opcodeParam(pCount-1);
         Item* msgs = &ctx->opcodeParam(pCount-2);
         for( int32 i = 1; i < pCount; ++i )
         {
            Item& msg = *msgs;
            ++msgs;
            if ( msg.isString() ) {
               delegate( instance, delegated, *msg.asString() );
            }
            else if (msg.isSymbol()) {
               delegate( instance, delegated, msg.asSymbol()->name() );
            }
            else {
               FALCON_RESIGN_XERROR( ParamError, e_inv_params, ctx,
                        .extra("Each delegated message must be a string or a symbol") );
               return;
            }
         }
      }

      ctx->popData(pCount);
      return;
   }

   BOM::handler handler = bom->get( message );
   if ( handler != 0  )
   {
      Function* func = static_cast<Function*>(Engine::instance()->getMantra(message));
      ctx->callInternal(func,pCount,ctx->opcodeParam(pCount));
   }
   else if( isOptional )
   {
      ctx->stackResult(pCount+1, Item());
   }
   else
   {
      op_summon_failing( ctx, instance, message, pCount );
   }
}


void Class::op_summon_failing( VMContext* ctx, void* instance, const String& message, int32 pCount ) const
{
   static const String message1 = OVERRIDE_OP_UNKMSG;

   Private::PropertyMap::iterator iter = _p->m_props.find( message1 );
   if( iter != _p->m_props.end() )
   {
      Property& prop = iter->second;
      Item temp = FALCON_GC_HANDLE( new String( message ) );
      // ok, even if pCount == 0
      ctx->insertData( pCount-1, &temp, 1, 0 );
      internal_callprop( this, ctx, instance, message1, pCount+1, prop );
      return;
   }

   FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                      .extra(message) );
}


void Class::op_getClassProperty( VMContext* ctx, const String& prop) const
{
   static BOM* bom = Engine::instance()->getBom();

   Private::PropertyMap::iterator iter = _p->m_props.find( prop );
   if( iter != _p->m_props.end() && iter->second.bStatic )
   {
      Property& prop = iter->second;
      if( prop.method != 0 ) {
         ctx->topData() = prop.method;
      }
      else if ( prop.bConst )
      {
         ctx->topData() = prop.value;
      }
      else {
         prop.getFunc( this, iter->first, 0, ctx->topData() );
      }
      return;
   }

   // try to find a valid BOM propery.
   BOM::handler handler = bom->get( prop );
   if ( handler != 0  )
   {
      // all bom methods do not modify their object
      // we can safely use a const cast
      handler( ctx, this->handler(), const_cast<Class*>(this) );
   }
   else
   {
      FALCON_RESIGN_XERROR( AccessTypeError, e_prop_acc, ctx,
                   .extra(prop) );
   }
}


void Class::op_setClassProperty( VMContext* ctx, const String& prop ) const
{
   FALCON_RESIGN_XERROR( AccessError, e_prop_acc, ctx,
                .extra(prop) );
}

void Class::op_isTrue( VMContext* ctx, void* ) const
{
   ctx->topData().setBoolean(true);
}


void Class::op_in( VMContext* ctx, void*) const
{
   ctx->topData().setBoolean(false);
}

void Class::op_provides( VMContext* ctx, void* instance, const String& propName ) const
{
   ctx->topData().setBoolean( hasProperty(instance, propName ) );
}


void Class::op_toString( VMContext* ctx, void *self ) const
{
   String *descr = new String();
   describe( self, *descr );
   ctx->stackResult(1, FALCON_GC_HANDLE(descr));
}


void Class::op_iter( VMContext* ctx, void* ) const
{
   Item item;
   item.setBreak();
   ctx->pushData(item);
}

void Class::op_next( VMContext* ctx, void* ) const
{
   Item item;
   item.setBreak();
   ctx->pushData(item);
}

}

/* end of class.cpp */
