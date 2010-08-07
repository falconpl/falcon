/**
 *  \file gtk_Misc.cpp
 */

#include "gtk_Misc.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Misc::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Misc = mod->addClass( "GtkMisc", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWidget" ) );
    c_Misc->getClassDef()->addInheritance( in );

    c_Misc->getClassDef()->factory( &Misc::factory );

    Gtk::MethodTab methods[] =
    {
    { "set_alignment",  &Misc::set_alignment },
    { "set_padding",    &Misc::set_padding },
    { "get_alignment",  &Misc::get_alignment },
    { "get_padding",    &Misc::get_padding },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Misc, meth->name, meth->cb );
}


Misc::Misc( const Falcon::CoreClass* gen, const GtkMisc* misc )
    :
    Gtk::CoreGObject( gen, (GObject*) misc )
{}


Falcon::CoreObject* Misc::factory( const Falcon::CoreClass* gen, void* misc, bool )
{
    return new Misc( gen, (GtkMisc*) misc );
}


/*#
    @class GtkMisc
    @brief Base class for widgets with alignments and padding.

    The GtkMisc widget is an abstract widget which is not useful itself, but is
    used to derive subclasses which have alignment and padding attributes.

    The horizontal and vertical padding attributes allows extra space to be added
    around the widget.

    The horizontal and vertical alignment attributes enable the widget to be
    positioned within its allocated area. Note that if the widget is added to a
    container in such a way that it expands automatically to fill its allocated
    area, the alignment settings will not alter the widgets position.
 */

/*#
    @method set_alignment GtkMisc
    @brief Sets the alignment of the widget.
    @param xalign the horizontal alignment, from 0 (left) to 1 (right). (float)
    @param yalign the vertical alignment, from 0 (top) to 1 (bottom). (float)
 */
FALCON_FUNC Misc::set_alignment( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || i_x->isNil() || !i_x->isOrdinal()
        || !i_y || i_y->isNil() || !i_y->isOrdinal() )
        throw_inv_params( "O,O" );
#endif
    gfloat x = i_x->asNumeric();
    gfloat y = i_y->asNumeric();
#ifndef NO_PARAMETER_CHECK
    if ( x < 0 || x > 1 || y < 0 || y > 1 )
        throw_inv_params( "0-1" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_misc_set_alignment( (GtkMisc*)_obj, x, y );
}


/*#
    @method set_padding GtkMisc
    @brief Sets the amount of space to add around the widget.
    @param xpad the amount of space to add on the left and right of the widget, in pixels.
    @param ypad the amount of space to add on the top and bottom of the widget, in pixels.
 */
FALCON_FUNC Misc::set_padding( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || i_x->isNil() || !i_x->isInteger()
        || !i_y || i_y->isNil() || !i_y->isInteger() )
        throw_inv_params( "I,I" );
#endif
    gint x = i_x->asInteger();
    gint y = i_y->asInteger();
    MYSELF;
    GET_OBJ( self );
    gtk_misc_set_alignment( (GtkMisc*)_obj, x, y );
}


/*#
    @method get_alignment GtkMisc
    @brief Gets the X and Y alignment of the widget within its allocation.
    @return [float x, float y]
 */
FALCON_FUNC Misc::get_alignment( VMARG )
{
    MYSELF;
    GET_OBJ( self );
    gfloat x, y;
    gtk_misc_get_alignment( (GtkMisc*)_obj, &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


/*#
    @method get_padding GtkMisc
    @brief Gets the padding in the X and Y directions of the widget.
    @return [int x, int y]
 */
FALCON_FUNC Misc::get_padding( VMARG )
{
    MYSELF;
    GET_OBJ( self );
    gint x, y;
    gtk_misc_get_padding( (GtkMisc*)_obj, &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( x );
    arr->append( y );
    vm->retval( arr );
}


} // Gtk
} // Falcon
