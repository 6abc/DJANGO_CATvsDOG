#!/bin/bash

# Quick Installation Script for Cat vs Dog Classifier Platform
# With Authentication, Model Sharing, and REST API

echo "🚀 Cat vs Dog Classifier - Complete Platform Setup"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python3 --version
if [ $? -ne 0 ]; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install --upgrade pip
pip install django djangorestframework tensorflow pillow numpy scipy

# Create Django project
echo -e "${BLUE}Creating Django project...${NC}"
django-admin startproject cat_dog_project .

# Create app
echo -e "${BLUE}Creating classifier app...${NC}"
python manage.py startapp classifier

# Create directory structure
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p classifier/templates/classifier
mkdir -p media/models
mkdir -p media/training_data
mkdir -p media/predictions
mkdir -p media/uploads
mkdir -p static/css

echo ""
echo -e "${GREEN}✅ Basic setup complete!${NC}"
echo ""
echo "📝 Next steps:"
echo "1. Copy all provided code files to their respective locations"
echo "2. Update cat_dog_project/settings.py with the provided configuration"
echo "3. Update cat_dog_project/urls.py with the provided configuration"
echo "4. Copy all templates to classifier/templates/classifier/"
echo "5. Run migrations:"
echo "   python manage.py makemigrations"
echo "   python manage.py migrate"
echo "6. Create superuser (optional):"
echo "   python manage.py createsuperuser"
echo "7. Run the server:"
echo "   python manage.py runserver"
echo ""
echo -e "${GREEN}📚 Documentation:${NC}"
echo "- README: See COMPLETE_README.md"
echo "- API Docs: See API_DOCUMENTATION.md"
echo ""
echo -e "${BLUE}🎉 Happy coding!${NC}"