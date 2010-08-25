/**
 *  \file g_ParamSpec.cpp
 */

#include "g_ParamSpec.hpp"


namespace Falcon {
namespace Glib {

/**
 *  \brief module init
 */
void ParamSpec::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ParamSpec = mod->addClass( "%GParamSpec" );

    c_ParamSpec->setWKS( true );
    c_ParamSpec->getClassDef()->factory( &ParamSpec::factory );

    mod->addClassProperty( c_ParamSpec, "name" );
    mod->addClassProperty( c_ParamSpec, "flags" );
    mod->addClassProperty( c_ParamSpec, "value_type" );
    mod->addClassProperty( c_ParamSpec, "owner_type" );
}


ParamSpec::ParamSpec( const Falcon::CoreClass* gen, const GParamSpec* spec )
    :
    Gtk::VoidObject( gen, spec )
{
    incref();
}


ParamSpec::ParamSpec( const ParamSpec& other )
    :
    Gtk::VoidObject( other )
{
    incref();
}


ParamSpec::~ParamSpec()
{
    decref();
}


void ParamSpec::incref() const
{
    if ( m_obj )
        g_param_spec_ref_sink( (GParamSpec*) m_obj );
}


void ParamSpec::decref() const
{
    if ( m_obj )
        g_param_spec_unref( (GParamSpec*) m_obj );
}


void ParamSpec::setObject( const void* spec )
{
    VoidObject::setObject( spec );
    incref();
}


bool ParamSpec::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    assert( m_obj );
    GParamSpec* m_spec = (GParamSpec*) m_obj;

    if ( s == "name" )
        it = UTF8String( m_spec->name );
    else
    if ( s == "flags" )
        it = (int64) m_spec->flags;
    else
    if ( s == "value_type" )
        it = (int64) m_spec->value_type;
    else
    if ( s == "owner_type" )
        it = (int64) m_spec->owner_type;
    else
        return false;
    return true;
}


bool ParamSpec::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* ParamSpec::factory( const Falcon::CoreClass* gen, void* spec, bool )
{
    return new ParamSpec( gen, (GParamSpec*) spec );
}


/*#
    @class GParamSpec
    @brief Metadata for parameter specifications
    @prop name name of this parameter
    @prop GParamFlags flags for this parameter
    @prop the GValue type for this parameter
    @prop GType type that uses (introduces) this paremeter

    GParamSpec is an object structure that encapsulates the metadata required to
    specify parameters, such as e.g. GObject properties.

    Parameter names need to start with a letter (a-z or A-Z). Subsequent characters
    can be letters, numbers or a '-'. All other characters are replaced by a '-'
    during construction. The result of this replacement is called the canonical
    name of the parameter.
 */


} // Glib
} // Falcon
