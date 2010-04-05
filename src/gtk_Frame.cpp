/**
 *  \file gtk_Frame.cpp
 */

#include "gtk_Frame.hpp"

#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Frame::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Frame = mod->addClass( "GtkFrame", &Frame::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Frame->getClassDef()->addInheritance( in );

    c_Frame->getClassDef()->factory( &Frame::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_label",          &Frame::set_label },
    { "set_label_widget",   &Frame::set_label_widget },
    { "set_label_align",    &Frame::set_label_align },
    { "set_shadow_type",    &Frame::set_shadow_type },
    { "get_label",          &Frame::get_label },
    { "get_label_align",    &Frame::get_label_align },
    { "get_label_widget",   &Frame::get_label_widget },
    { "get_shadow_type",    &Frame::get_shadow_type },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Frame, meth->name, meth->cb );
}


Frame::Frame( const Falcon::CoreClass* gen, const GtkFrame* frm )
    :
    Gtk::CoreGObject( gen, (GObject*) frm )
{}


Falcon::CoreObject* Frame::factory( const Falcon::CoreClass* gen, void* frm, bool )
{
    return new Frame( gen, (GtkFrame*) frm );
}


/*#
    @class GtkFrame
    @brief A bin with a decorative frame and optional label
    @optparam label the text to use as the label of the frame

    The frame widget is a Bin that surrounds its child with a decorative frame and
    an optional label. If present, the label is drawn in a gap in the top side of
    the frame. The position of the label can be controlled with gtk_frame_set_label_align().
 */
FALCON_FUNC Frame::init( VMARG )
{
    MYSELF;

    if ( self->getGObject() )
        return;

    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_lbl && ( i_lbl->isNil() || !i_lbl->isString() ) )
        throw_inv_params( "[S]" );
#endif
    GtkWidget* wdt;
    if ( i_lbl )
    {
        AutoCString lbl( *i_lbl );
        wdt = gtk_frame_new( lbl.c_str() );
    }
    else
        wdt = gtk_frame_new( NULL );

    self->setGObject( (GObject*) wdt );
}


/*#
    @method set_label GtkFrame
    @brief Sets the text of the label. If label is nil, the current label is removed.
    @param the text to use as the label of the frame (or nil)
 */
FALCON_FUNC Frame::set_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_lbl && ( i_lbl->isNil() || !i_lbl->isString() ) )
        throw_inv_params( "[S]" );
#endif
    MYSELF;
    GET_OBJ( self );
    if ( i_lbl )
    {
        AutoCString lbl( *i_lbl );
        gtk_frame_set_label( (GtkFrame*)_obj, lbl.c_str() );
    }
    else
        gtk_frame_set_label( (GtkFrame*)_obj, NULL );
}


/*#
    @method set_label_widget GtkFrame
    @brief Sets the label widget for the frame.
    @param label_widget the new label widget

    This is the widget that will appear embedded in the top edge of the frame as a title.
 */
FALCON_FUNC Frame::set_label_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_frame_set_label_widget( (GtkFrame*)_obj, wdt );
}


/*#
    @method set_label_align GtkFrame
    @brief Sets the alignment of the frame widget's label.
    @param xalign The position of the label along the top edge of the widget. A value of 0.0 represents left alignment; 1.0 represents right alignment.
    @param yalign The y alignment of the label. A value of 0.0 aligns under the frame; 1.0 aligns above the frame. If the values are exactly 0.0 or 1.0 the gap in the frame won't be painted because the label will be completely above or below the frame.

    The default values for a newly created frame are 0.0 and 0.5.
 */
FALCON_FUNC Frame::set_label_align( VMARG )
{
    Item* i_xalign = vm->param( 0 );
    Item* i_yalign = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_xalign || i_xalign->isNil() || !i_xalign->isOrdinal()
        || !i_yalign || i_yalign->isNil() || !i_yalign->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_frame_set_label_align( (GtkFrame*)_obj, i_xalign->asNumeric(), i_yalign->asNumeric() );
}


/*#
    @method set_shadow_type GtkFrame
    @brief Sets the shadow type for frame.
    @param type (GtkShadowType) the new GtkShadowType
 */
FALCON_FUNC Frame::set_shadow_type( VMARG )
{
    Item* i_shad = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_shad || i_shad->isNil() || !i_shad->isInteger() )
        throw_inv_params( "GtkShadowType" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_frame_set_shadow_type( (GtkFrame*)_obj, (GtkShadowType) i_shad->asInteger() );
}


/*#
    @method get_label GtkFrame
    @brief If the frame's label widget is a GtkLabel, returns the text in the label widget, else return nil
    @return the text in the label, or nil if there was no label widget or the lable widget was not a GtkLabel.

    (The frame will have a GtkLabel for the label widget if a non-NULL argument was passed to gtk_frame_new().)
 */
FALCON_FUNC Frame::get_label( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_frame_get_label( (GtkFrame*)_obj );
    if ( lbl )
    {
        String* s = new String( lbl );
        s->bufferize();
        vm->retval( s );
    }
    else
        vm->retnil();
}


/*#
    @method get_label_align GtkFrame
    @brief Retrieves the X and Y alignment of the frame's label.
    @return [ xalign, yalign ]
 */
FALCON_FUNC Frame::get_label_align( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gfloat xalign, yalign;
    gtk_frame_get_label_align( (GtkFrame*)_obj, &xalign, &yalign );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( xalign );
    arr->append( yalign );
    vm->retval( arr );
}


/*#
    @method get_label_widget GtkFrame
    @brief Retrieves the label widget for the frame.
    @return the label widget, or nil if there is none.
 */
FALCON_FUNC Frame::get_label_widget( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_frame_get_label_widget( (GtkFrame*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method get_shadow_type GtkFrame
    @brief Retrieves the shadow type of the frame.
    @return (GtkShadowType) the current shadow type of the frame.
 */
FALCON_FUNC Frame::get_shadow_type( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_frame_get_shadow_type( (GtkFrame*)_obj ) );
}


} // Gtk
} // Falcon
