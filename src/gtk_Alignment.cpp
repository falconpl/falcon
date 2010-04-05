/**
 *  \file gtk_Alignment.cpp
 */

#include "gtk_Alignment.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Alignment::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Alignment = mod->addClass( "GtkAlignment", &Alignment::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Alignment->getClassDef()->addInheritance( in );

    c_Alignment->getClassDef()->factory( &Alignment::factory );

    mod->addClassMethod( c_Alignment, "set",         &Alignment::set );
    mod->addClassMethod( c_Alignment, "get_padding", &Alignment::get_padding );
    mod->addClassMethod( c_Alignment, "set_padding", &Alignment::set_padding );
}


Alignment::Alignment( const Falcon::CoreClass* gen, const GtkAlignment* alignment )
    :
    Gtk::CoreGObject( gen, (GObject*) alignment )
{}


Falcon::CoreObject* Alignment::factory( const Falcon::CoreClass* gen, void* alignment, bool )
{
    return new Alignment( gen, (GtkAlignment*) alignment );
}


/*#
    @class GtkAlignment
    @brief A widget which controls the alignment and size of its child
    @optparam xalign the horizontal alignment of the child widget, from 0 (left) to 1 (right).
    @optparam yalign the vertical alignment of the child widget, from 0 (top) to 1 (bottom).
    @optparam xscale the amount that the child widget expands horizontally to fill up unused space, from 0 to 1. A value of 0 indicates that the child widget should never expand. A value of 1 indicates that the child widget will expand to fill all of the space allocated for the GtkAlignment.
    @optparam yscale the amount that the child widget expands vertically to fill up unused space, from 0 to 1. The values are similar to xscale.

    The GtkAlignment widget controls the alignment and size of its child widget.
    It has four settings: xscale, yscale, xalign, and yalign.

    The scale settings are used to specify how much the child widget should expand
    to fill the space allocated to the GtkAlignment. The values can range from 0
    (meaning the child doesn't expand at all) to 1 (meaning the child expands to
    fill all of the available space).

    The align settings are used to place the child widget within the available
    area. The values range from 0 (top or left) to 1 (bottom or right). Of course,
    if the scale settings are both set to 1, the alignment settings have no effect.
 */
FALCON_FUNC Alignment::init( VMARG )
{
    Gtk::ArgCheck0 args( vm, "N,N,N,N" );

    gfloat xalign = args.getNumeric( 0, false );
    gfloat yalign = args.getNumeric( 1, false );
    gfloat xscale = args.getNumeric( 2, false );
    gfloat yscale = args.getNumeric( 3, false );

    MYSELF;
    GtkWidget* wdt = gtk_alignment_new( xalign, yalign, xscale, yscale );
    self->setGObject( (GObject*) wdt );
}


/*#
    @method set GtkAlignment
    @brief Sets the GtkAlignment values.
    @param xalign
    @param yalign
    @param xscale
    @param yscale
 */
FALCON_FUNC Alignment::set( VMARG )
{
    Gtk::ArgCheck0 args( vm, "N,N,N,N" );

    gfloat xalign = args.getNumeric( 0 );
    gfloat yalign = args.getNumeric( 1 );
    gfloat xscale = args.getNumeric( 2 );
    gfloat yscale = args.getNumeric( 3 );

    MYSELF;
    GET_OBJ( self );
    gtk_alignment_set( (GtkAlignment*)_obj, xalign, yalign, xscale, yscale );
}


/*#
    @method get_padding GtkAlignment
    @brief Gets the padding on the different sides of the widget.
    @return [ top, bottom, left, right ]
 */
FALCON_FUNC Alignment::get_padding( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    guint top, bottom, left, right;
    gtk_alignment_get_padding( (GtkAlignment*)_obj, &top, &bottom, &left, &right );
    CoreArray* arr = new CoreArray( 4 );
    arr->append( top );
    arr->append( bottom );
    arr->append( left );
    arr->append( right );
    vm->retval( arr );
}


/*#
    @method set_padding GtkAlignment
    @brief Sets the padding on the different sides of the widget.
    @param padding_top the padding at the top of the widget
    @param padding_bottom the padding at the bottom of the widget
    @param padding_left the padding at the left of the widget
    @param padding_right the padding at the right of the widget.

    The padding adds blank space to the sides of the widget. For instance, this
    can be used to indent the child widget towards the right by adding padding on the left.
 */
FALCON_FUNC Alignment::set_padding( VMARG )
{
    Gtk::ArgCheck0 args( vm, "I,I,I,I" );

    gfloat pad_top = args.getInteger( 0 );
    gfloat pad_bot = args.getInteger( 1 );
    gfloat pad_lef = args.getInteger( 2 );
    gfloat pad_rig = args.getInteger( 3 );

    MYSELF;
    GET_OBJ( self );
    gtk_alignment_set_padding( (GtkAlignment*)_obj, pad_top, pad_bot, pad_lef, pad_rig );
}


} // Gtk
} // Falcon
