/**
 *  \file gdk_DragContext.cpp
 */

#include "gdk_DragContext.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gdk {

/**
 *  \brief module init
 */
void DragContext::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_DragContext = mod->addClass( "GdkDragContext", &Gtk::abstract_init );

    c_DragContext->setWKS( true );
    c_DragContext->getClassDef()->factory( &DragContext::factory );

    //mod->addClassProperty( c_DragContext, "parent_instance" );
    mod->addClassProperty( c_DragContext, "protocol" );
    mod->addClassProperty( c_DragContext, "is_source" );
    //mod->addClassProperty( c_DragContext, "source_window" );
    //mod->addClassProperty( c_DragContext, "dest_window" );
    //mod->addClassProperty( c_DragContext, "targets" );
    mod->addClassProperty( c_DragContext, "actions" );
    mod->addClassProperty( c_DragContext, "suggested_action" );
    mod->addClassProperty( c_DragContext, "action" );
    //mod->addClassProperty( c_DragContext, "time" );
}


DragContext::DragContext( const Falcon::CoreClass* gen, const GdkDragContext* ctxt )
    :
    Falcon::CoreObject( gen ),
    m_ctxt( NULL )
{
    if ( ctxt )
    {
        m_ctxt = (GdkDragContext*) ctxt;
        gdk_drag_context_ref( m_ctxt );
    }
}


DragContext::~DragContext()
{
    if ( m_ctxt )
        gdk_drag_context_unref( m_ctxt );
}


bool DragContext::getProperty( const Falcon::String& s, Falcon::Item& it ) const
{
    if ( s == "protocol" )
        it = m_ctxt->protocol;
    else
    if ( s == "is_source" )
        it = (bool) m_ctxt->is_source;
    else
    if ( s == "actions" )
        it = m_ctxt->actions;
    else
    if ( s == "suggested_action" )
        it = m_ctxt->suggested_action;
    else
    if ( s == "action" )
        it = m_ctxt->action;
    else
        return false;
    return true;
}


bool DragContext::setProperty( const Falcon::String& s, const Falcon::Item& it )
{
    return false;
}


Falcon::CoreObject* DragContext::factory( const Falcon::CoreClass* gen, void* ctxt, bool )
{
    return new DragContext( gen, (GdkDragContext*) ctxt );
}


/*#
    @class GdkDragContext
    @brief A GdkDragContext holds information about a drag in progress.

    It is used on both source and destination sides.

    @prop parent_instance TODO (GObject*) the parent instance
    @prop protocol the DND protocol which governs this drag.
    @prop is_source true if the context is used on the source side.
    @prop source_window TODO the source of this drag.
    @prop dest_window TODO the destination window of this drag.
    @prop targets TODO a list of targets offered by the source.
    @prop actions a bitmask of actions proposed by the source when suggested_action is GDK_ACTION_ASK.
    @prop suggested_action the action suggested by the source.
    @prop action the action chosen by the destination.
    @prop start_time a timestamp recording the start time of this drag.
 */


} // Gdk
} // Falcon
