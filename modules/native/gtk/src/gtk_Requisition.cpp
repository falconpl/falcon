/**
 *  \file gtk_Requisition.cpp
 */

#include "gtk_Requisition.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Requisition::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Requisition = mod->addClass( "GtkRequisition", &Requisition::init )
        ->addParam( "width" )->addParam( "height" );

    c_Requisition->setWKS( true );
    c_Requisition->getClassDef()->factory( &Requisition::factory );

    mod->addClassProperty( c_Requisition, "width" );
    mod->addClassProperty( c_Requisition, "height" );
}


Requisition::Requisition( const Falcon::CoreClass* gen, const GtkRequisition* req )
    :
    Falcon::CoreObject( gen )
{
    GtkRequisition* m_req = (GtkRequisition*) memAlloc( sizeof( GtkRequisition ) );

    if ( !req )
    {
        m_req->width = 0;
        m_req->height = 0;
    }
    else
    {
        m_req->width = req->width;
        m_req->height = req->height;
    }

    setUserData( m_req );
}


Requisition::~Requisition()
{
    GtkRequisition* req = (GtkRequisition*) getUserData();
    if ( req )
        memFree( req );
}


bool Requisition::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    GtkRequisition* m_req = (GtkRequisition*) getUserData();

    if ( s == "width" )
        it = m_req->width;
    else
    if ( s == "height" )
        it = m_req->height;
    else
        return false;
    return true;
}


bool Requisition::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    GtkRequisition* m_req = (GtkRequisition*) getUserData();

    if ( s == "width" )
        m_req->width = it.forceInteger();
    else
    if ( s == "height" )
        m_req->height = it.forceInteger();
    else
        return false;
    return true;
}


Falcon::CoreObject* Requisition::factory( const Falcon::CoreClass* gen, void* req, bool )
{
    return new Requisition( gen, (GtkRequisition*) req );
}


/*#
    @class GtkRequisition
    @brief A GtkRequisition represents the desired size of a widget.
    @optparam width
    @optparam height

    @prop width the widget's desired width
    @prop heigth the widget's desired height
 */
FALCON_FUNC Requisition::init( VMARG )
{
    Item* i_w = vm->param( 0 );
    Item* i_h = vm->param( 1 );
    int w = 0, h = 0;
    if ( i_w )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_w->isNil() || !i_w->isInteger() )
            throw_inv_params( "[I,I]" );
#endif
        w = i_w->asInteger();
    }
    if ( i_h )
    {
#ifndef NO_PARAMETER_CHECK
        if ( i_h->isNil() || !i_h->isInteger() )
            throw_inv_params( "[I,I]" );
#endif
        h = i_h->asInteger();
    }
    MYSELF;
    GtkRequisition* req = (GtkRequisition*) self->getUserData();
    req->width = w;
    req->height = h;
}


} // Gtk
} // Falcon
