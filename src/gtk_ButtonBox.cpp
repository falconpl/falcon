/**
 *  \file gtk_ButtonBox.cpp
 */

#include "gtk_ButtonBox.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ButtonBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ButtonBox = mod->addClass( "ButtonBox", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "Box" ) );
    c_ButtonBox->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    //{ "get_layout",             &ButtonBox::get_layout },
    { "get_child_size",         &ButtonBox::get_child_size },
    { "get_child_ipadding",     &ButtonBox::get_child_ipadding },
    { "get_child_secondary",    &ButtonBox::get_child_secondary },
    //{ "set_spacing",            &ButtonBox::set_spacing },
    //{ "set_layout",             &ButtonBox::set_layout },
    //{ "set_child_size",         &ButtonBox::set_child_size },
    //{ "set_child_ipadding",     &ButtonBox::set_child_ipadding },
    { "set_child_secondary",    &ButtonBox::set_child_secondary },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ButtonBox, meth->name, meth->cb );
}

/*#
    @class gtk.ButtonBox
    @brief Abstract container class.

    The primary purpose of this class is to keep track of the various properties
    of GtkHButtonBox and GtkVButtonBox widgets.
 */


//FALCON_FUNC ButtonBox::get_layout( VMARG );


/*#
    @method get_child_size gtk.ButtonBox
    @brief Retrieves the current width and height of all child widgets in a button box.
    @return Array( minimum width, minimum height )
 */
FALCON_FUNC ButtonBox::get_child_size( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    gint minw, minh;
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_get_child_size( (GtkButtonBox*)_obj, &minw, &minh );
    CoreArray* arr = new CoreArray;
    arr->append( minw );
    arr->append( minh );
    vm->retval( arr );
}


/*#
    @method get_child_ipadding gtk.ButtonBox
    @brief Gets the default number of pixels that pad the buttons in a given button box.
    @return Array( padx, pady )

    get_child_ipadding is deprecated and should not be used in newly-written code.
    Use the style properties "child-internal-pad-x" and "child-internal-pad-y" instead.
 */
FALCON_FUNC ButtonBox::get_child_ipadding( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    gint w, h;
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_get_child_ipadding( (GtkButtonBox*)_obj, &w, &h );
    CoreArray* arr = new CoreArray;
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method get_child_secondary gtk.ButtonBox
    @brief Returns whether child should appear in a secondary group of children.
    @param child (widget)
    @return (boolean)
 */
FALCON_FUNC ButtonBox::get_child_secondary( VMARG )
{
    Item* i_child = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !( i_child->isOfClass( "Widget" ) || i_child->isOfClass( "gtk.Widget" ) ) )
        throw_inv_params( "Widget" );
#endif
    GtkWidget* child = (GtkWidget*)((GData*)i_child->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gboolean b = gtk_button_box_get_child_secondary( (GtkButtonBox*)_obj, child );
    vm->retval( (bool) b );
}


//FALCON_FUNC ButtonBox::set_spacing( VMARG )

//FALCON_FUNC ButtonBox::set_layout( VMARG );

//FALCON_FUNC ButtonBox::set_child_size( VMARG );

//FALCON_FUNC ButtonBox::set_child_ipadding( VMARG );


/*#
    @method set_child_secondary gtk.ButtonBox
    @brief Sets whether child should appear in a secondary group of children.
    @param widget
    @param is_secondary (boolean)

    A typical use of a secondary child is the help button in a dialog.
 */
FALCON_FUNC ButtonBox::set_child_secondary( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_sec = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !( i_child->isOfClass( "Widget" ) || i_child->isOfClass( "gtk.Widget" ) )
        || !i_sec || i_sec->isNil() || !i_sec->isBoolean() )
        throw_inv_params( "Widget,B" );
#endif
    GtkWidget* child = (GtkWidget*)((GData*)i_child->asObject()->getUserData())->obj();
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_set_child_secondary( (GtkButtonBox*)_obj, child,
        i_sec->asBoolean() ? TRUE : FALSE );
}


} // Gtk
} // Falcon
