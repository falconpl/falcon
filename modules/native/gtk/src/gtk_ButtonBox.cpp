/**
 *  \file gtk_ButtonBox.cpp
 */

#include "gtk_ButtonBox.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ButtonBox::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ButtonBox = mod->addClass( "GtkButtonBox", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBox" ) );
    c_ButtonBox->getClassDef()->addInheritance( in );

    c_ButtonBox->getClassDef()->factory( &ButtonBox::factory );

    Gtk::MethodTab methods[] =
    {
#if 0 // deprecated
    { "get_spacing",            &ButtonBox::get_spacing },
#endif
    { "get_layout",             &ButtonBox::get_layout },
    { "get_child_size",         &ButtonBox::get_child_size },
    { "get_child_ipadding",     &ButtonBox::get_child_ipadding },
    { "get_child_secondary",    &ButtonBox::get_child_secondary },
#if 0 // deprecated
    { "set_spacing",            &ButtonBox::set_spacing },
#endif
    { "set_layout",             &ButtonBox::set_layout },
#if 0 // deprecated
    { "set_child_size",         &ButtonBox::set_child_size },
    { "set_child_ipadding",     &ButtonBox::set_child_ipadding },
#endif
    { "set_child_secondary",    &ButtonBox::set_child_secondary },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ButtonBox, meth->name, meth->cb );
}


ButtonBox::ButtonBox( const Falcon::CoreClass* gen, const GtkButtonBox* box )
    :
    Gtk::CoreGObject( gen, (GObject*) box )
{}

Falcon::CoreObject* ButtonBox::factory( const Falcon::CoreClass* gen, void* box, bool )
{
    return new ButtonBox( gen, (GtkButtonBox*) box );
}


/*#
    @class GtkButtonBox
    @brief Base class for GtkHButtonBox and GtkVButtonBox

    The primary purpose of this class is to keep track of the various properties
    of GtkHButtonBox and GtkVButtonBox widgets.

    gtk_button_box_get_child_size() retrieves the minimum width and height for
    widgets in a given button box.

    The internal padding of buttons can be retrieved and changed per button box
    using gtk_button_box_get_child_ipadding() and
    gtk_button_box_set_child_ipadding() respectively.

    gtk_button_box_get_spacing() and gtk_button_box_set_spacing() retrieve
    and change default number of pixels between buttons, respectively.

    gtk_button_box_get_layout() and gtk_button_box_set_layout() retrieve and
    alter the method used to spread the buttons in a button box across the
    container, respectively.

    The main purpose of GtkButtonBox is to make sure the children have all
    the same size. Therefore it ignores the homogeneous property which it
    inherited from GtkBox, and always behaves as if homogeneous was TRUE.
 */


/*#
    @method get_layout GtkButtonBox
    @brief Retrieves the method being used to arrange the buttons in a button box.
    @return the method used to layout buttons in widget (GtkButtonBoxStyle).
 */
FALCON_FUNC ButtonBox::get_layout( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_button_box_get_layout( (GtkButtonBox*)_obj ) );
}


/*#
    @method get_child_size GtkButtonBox
    @brief Retrieves the current width and height of all child widgets in a button box.
    @return an array ( minimum width, minimum height ).
 */
FALCON_FUNC ButtonBox::get_child_size( VMARG )
{
    NO_ARGS
    gint minw, minh;
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_get_child_size( (GtkButtonBox*)_obj, &minw, &minh );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( minw );
    arr->append( minh );
    vm->retval( arr );
}


/*#
    @method get_child_ipadding GtkButtonBox
    @brief Gets the default number of pixels that pad the buttons in a given button box.
    @return an array ( pad_x, pad_y ).

    get_child_ipadding is deprecated and should not be used in newly-written code.
    Use the style properties "child-internal-pad-x" and "child-internal-pad-y" instead.
 */
FALCON_FUNC ButtonBox::get_child_ipadding( VMARG )
{
    NO_ARGS
    gint w, h;
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_get_child_ipadding( (GtkButtonBox*)_obj, &w, &h );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( w );
    arr->append( h );
    vm->retval( arr );
}


/*#
    @method get_child_secondary GtkButtonBox
    @brief Returns whether child should appear in a secondary group of children.
    @param child a child of widget
    @return whether child should appear in a secondary group of children.
 */
FALCON_FUNC ButtonBox::get_child_secondary( VMARG )
{
    Item* i_child = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !i_child->isObject() || !IS_DERIVED( i_child, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_box_get_child_secondary( (GtkButtonBox*)_obj, child ) );
}


#if 0 // deprecated
FALCON_FUNC ButtonBox::set_spacing( VMARG )
#endif


/*#
    @method set_layout GtkButtonBox
    @brief Changes the way buttons are arranged in their container.
    @param layout_style the new layout style (GtkButtonBoxStyle).
 */
FALCON_FUNC ButtonBox::set_layout( VMARG )
{
    Item* i_style = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_style || !i_style->isInteger() )
        throw_inv_params( "GtkButtonBoxStyle" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_set_layout( (GtkButtonBox*)_obj,
                               (GtkButtonBoxStyle) i_style->asInteger() );
}


#if 0 // deprecated
FALCON_FUNC ButtonBox::set_child_size( VMARG );
FALCON_FUNC ButtonBox::set_child_ipadding( VMARG );
#endif


/*#
    @method set_child_secondary GtkButtonBox
    @brief Sets whether child should appear in a secondary group of children.
    @param child a child of widget
    @param is_secondary if TRUE, the child appears in a secondary group of the button box.

    A typical use of a secondary child is the help button in a dialog.

    This group appears after the other children if the style is GTK_BUTTONBOX_START,
    GTK_BUTTONBOX_SPREAD or GTK_BUTTONBOX_EDGE, and before the other children if
    the style is GTK_BUTTONBOX_END. For horizontal button boxes, the definition
    of before/after depends on direction of the widget (see gtk_widget_set_direction()).
    If the style is GTK_BUTTONBOX_START or GTK_BUTTONBOX_END, then the secondary
    children are aligned at the other end of the button box from the main children.
    For the other styles, they appear immediately next to the main children.
 */
FALCON_FUNC ButtonBox::set_child_secondary( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_sec = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || !i_child->isObject() || !IS_DERIVED( i_child, GtkWidget )
        || !i_sec || !i_sec->isBoolean() )
        throw_inv_params( "GtkWidget,B" );
#endif
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_button_box_set_child_secondary( (GtkButtonBox*)_obj, child,
                                        (gboolean) i_sec->asBoolean() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
